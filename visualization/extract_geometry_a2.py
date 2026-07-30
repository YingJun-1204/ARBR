"""Extract sample-adaptive Gaussian geometry for Experiment 5.4 A1.

Key fixes relative to the original script:
1. Reuses the saved run configuration when available instead of silently rebuilding
   every checkpoint from one hard-coded configuration.
2. Loads checkpoints strictly by default and reports configuration/state mismatches.
3. Runs the real model forward and captures the generator/router inputs and outputs
   with forward hooks. The hard gate is taken from the router's actual forward value.
4. Uses replicate-padded smoothing for the fallback complexity statistic.
5. Uses all test windows by default and supports fixed-channel sample selection.
6. Saves enough metadata to make every selected example reproducible.

For the most reliable paper result, save a config.json/args.json next to each
checkpoint and expose the actual constrained geometry from the model forward as a
mapping named ``last_geometry`` (or return a dictionary containing mu/sigma/alpha).
If the model does not expose constrained tensors, this script falls back to the
paper parameterization sigmoid(mu), softplus(sigma), sigmoid(alpha) and emits a
warning in the manifest.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Ensure the project root is importable.
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_provider.data_factory import data_provider  # noqa: E402
from models.gs_linear import Model  # noqa: E402


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "freq": "h"},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "freq": "h"},
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "freq": "t"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "freq": "t"},
    "weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "freq": "h"},
    "electricity": {"data": "custom", "data_path": "electricity.csv", "enc_in": 321, "freq": "h"},
    "traffic": {"data": "custom", "data_path": "traffic.csv", "enc_in": 862, "freq": "h"},
}

# Paper-reported fallback values. Saved training config takes precedence.
K_BASE_FALLBACK = {
    "ETTh1": 1,
    "ETTh2": 2,
    "ETTm1": 3,
    "ETTm2": 2,
    "weather": 3,
    "electricity": 1,
    "traffic": 1,
}

CONFIG_FILENAMES = (
    "config.json",
    "args.json",
    "model_config.json",
    "hyperparameters.json",
    "hparams.json",
)


# -----------------------------------------------------------------------------
# Configuration and checkpoint loading
# -----------------------------------------------------------------------------

def _fallback_config(
    dataset_name: str,
    pred_len: int,
    seq_len: int,
    root_path: str,
    checkpoints_dir: str,
) -> Dict[str, Any]:
    """Fallback only; a saved training configuration is preferred."""
    ds_cfg = DATASET_CONFIGS.get(
        dataset_name,
        {"data": "custom", "data_path": f"{dataset_name}.csv", "enc_in": 7, "freq": "h"},
    )
    return {
        "task_name": "long_term_forecast",
        "is_training": 0,
        "model_id": f"{dataset_name.lower()}_jet_{pred_len}",
        "model": "SplatTS",
        "data": ds_cfg["data"],
        "root_path": root_path,
        "data_path": ds_cfg["data_path"],
        "features": "M",
        "target": "OT",
        "freq": ds_cfg["freq"],
        "checkpoints": checkpoints_dir,
        "seq_len": seq_len,
        "label_len": 48,
        "pred_len": pred_len,
        "enc_in": ds_cfg["enc_in"],
        "d_model": 128,
        "representation": "gs",
        "patch_len": 24,
        "stride": 12,
        "head_dropout": 0.1,
        "head_dropout_position": "post",
        "head_mode": "linear",
        "gs_dropout": 0.3,
        "gs_weight_decay": 1e-4,
        "num_gaussians": 8,
        "density_mode": "cas",
        "use_occlusion": False,
        "use_residual": True,
        "gs_residual_weight": 0.1,
        "k_base": K_BASE_FALLBACK.get(dataset_name, -1),
        "output_dir": "loss_cas_simplify",
        "ablation_mode": "none",
        "num_implicit_gaussians": 4,
        "jet_max_shift_samples": 1.0,
        "jet_score_temperature": 0.01,
        "jet_density_tau": 1.0,
        "jet_detach_geometry": 1,
        "jet_scale_init": 0.1,
        "jet_sigma_init": 0.2,
        "jet_derivative_mode": "centered_legacy",
        "fusion_mode": "geometry",
        "fusion_hidden_dim": 16,
        "fusion_init": 0.5,
        "fusion_beta_max": 0.75,
        "fusion_detach_geometry": 1,
        "num_workers": 0,
        "batch_size": 64,
        "use_gpu": torch.cuda.is_available(),
        "gpu": 0,
        "use_multi_gpu": False,
    }


def find_checkpoint_path(
    checkpoints_dir: str,
    dataset_name: str,
    pred_len: int,
) -> Optional[Path]:
    root = Path(checkpoints_dir)
    possible_names = (
        f"long_term_forecast_{dataset_name.lower()}_jet_{pred_len}_SplatTS",
        f"long_term_forecast_{dataset_name}_jet_{pred_len}_SplatTS",
    )
    for name in possible_names:
        path = root / name / "checkpoint.pth"
        if path.exists():
            return path

    if root.exists():
        matches: List[Path] = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            if dataset_name.lower() in folder.name.lower() and f"_{pred_len}_" in folder.name:
                candidate = folder / "checkpoint.pth"
                if candidate.exists():
                    matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple matching checkpoints were found. Pass a more specific "
                f"--checkpoints directory. Candidates: {[str(p) for p in matches]}"
            )
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    # Some training scripts save {"args": {...}} or {"config": {...}}.
    for key in ("args", "config", "hparams", "hyperparameters"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            return dict(nested)
    return dict(obj)


def find_saved_config(run_dir: Path, explicit_config: Optional[str]) -> Optional[Path]:
    if explicit_config:
        path = Path(explicit_config)
        if not path.exists():
            raise FileNotFoundError(f"Explicit config does not exist: {path}")
        return path
    for filename in CONFIG_FILENAMES:
        path = run_dir / filename
        if path.exists():
            return path
    return None


def build_args(
    dataset_name: str,
    pred_len: int,
    seq_len: int,
    root_path: str,
    checkpoints_dir: str,
    checkpoint_path: Path,
    explicit_config: Optional[str],
    require_saved_config: bool,
) -> Tuple[argparse.Namespace, str]:
    fallback = _fallback_config(dataset_name, pred_len, seq_len, root_path, checkpoints_dir)
    config_path = find_saved_config(checkpoint_path.parent, explicit_config)

    if config_path is not None:
        saved = _read_json(config_path)
        merged = {**fallback, **saved}
        config_source = str(config_path)
    else:
        if require_saved_config:
            raise FileNotFoundError(
                f"No saved config was found next to {checkpoint_path}. "
                "Provide --config or save config.json/args.json in the run directory."
            )
        merged = fallback
        config_source = "fallback_defaults"
        warnings.warn(
            f"No saved training config found for {dataset_name} H={pred_len}. "
            "Using fallback defaults. This is suitable for debugging only; final paper "
            "geometry should be extracted with the exact training configuration.",
            RuntimeWarning,
        )

    # Runtime/data-location fields must match this extraction invocation.
    ds_cfg = DATASET_CONFIGS.get(
        dataset_name,
        {"data": "custom", "data_path": f"{dataset_name}.csv", "enc_in": merged.get("enc_in", 7), "freq": "h"},
    )
    merged.update(
        {
            "task_name": merged.get("task_name", "long_term_forecast"),
            "is_training": 0,
            "model_id": merged.get("model_id", f"{dataset_name.lower()}_jet_{pred_len}"),
            "data": ds_cfg["data"],
            "root_path": root_path,
            "data_path": ds_cfg["data_path"],
            "freq": ds_cfg["freq"],
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": ds_cfg["enc_in"],
            "checkpoints": checkpoints_dir,
            "use_gpu": torch.cuda.is_available(),
            "use_multi_gpu": False,
            "num_workers": int(merged.get("num_workers", 0)),
        }
    )
    if int(merged.get("k_base", -1)) < 0 and dataset_name in K_BASE_FALLBACK:
        merged["k_base"] = K_BASE_FALLBACK[dataset_name]
    return argparse.Namespace(**merged), config_source


def _extract_state_dict(checkpoint_obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint_obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint_obj.get(key)
            if isinstance(value, Mapping) and value:
                return value
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise TypeError(
        "Unsupported checkpoint format. Expected a state_dict or a mapping containing "
        "state_dict/model_state_dict/model."
    )


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    allow_non_strict: bool,
) -> Dict[str, List[str]]:
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint_obj))

    if not allow_non_strict:
        model.load_state_dict(state_dict, strict=True)
        return {"missing_keys": [], "unexpected_keys": []}

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        warnings.warn(
            "Non-strict checkpoint loading produced mismatches. "
            f"Missing={missing}; unexpected={unexpected}",
            RuntimeWarning,
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected}


# -----------------------------------------------------------------------------
# Forward capture and geometry decoding
# -----------------------------------------------------------------------------

def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


def _find_mapping_with_geometry(value: Any) -> Optional[Mapping[str, Any]]:
    """Recursively find a dictionary containing geometry keys."""
    if isinstance(value, Mapping):
        lower = {str(key).lower(): key for key in value.keys()}
        if "mu" in lower and "sigma" in lower and "alpha" in lower:
            return value
        for nested in value.values():
            found = _find_mapping_with_geometry(nested)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_mapping_with_geometry(nested)
            if found is not None:
                return found
    return None


def _geometry_from_module_cache(module: torch.nn.Module) -> Optional[Mapping[str, Any]]:
    for attr in ("last_geometry", "geometry_cache", "cached_geometry", "_last_geometry"):
        value = getattr(module, attr, None)
        found = _find_mapping_with_geometry(value)
        if found is not None:
            return found
    return None


def _normalize_geometry_mapping(
    mapping: Mapping[str, Any],
    expected_rows: int,
    expected_k: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    lower = {str(key).lower(): key for key in mapping.keys()}
    try:
        mu = mapping[lower["mu"]]
        sigma = mapping[lower["sigma"]]
        alpha = mapping[lower["alpha"]]
    except KeyError:
        return None
    if not all(torch.is_tensor(tensor) for tensor in (mu, sigma, alpha)):
        return None

    tensors = []
    for tensor in (mu, sigma, alpha):
        tensor = tensor.detach()
        if tensor.numel() != expected_rows * expected_k:
            return None
        tensors.append(tensor.reshape(expected_rows, expected_k))
    return tensors[0], tensors[1], tensors[2]


def _build_decoder_input(
    batch_y: torch.Tensor,
    pred_len: int,
    label_len: int,
) -> torch.Tensor:
    batch_size, _, channels = batch_y.shape
    zeros = torch.zeros(batch_size, pred_len, channels, device=batch_y.device, dtype=batch_y.dtype)
    if label_len > 0 and batch_y.shape[1] >= label_len:
        return torch.cat([batch_y[:, :label_len, :], zeros], dim=1)
    return zeros


def _run_full_forward(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: Optional[torch.Tensor],
    batch_y_mark: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> Any:
    """Call the model using common forecasting signatures without swallowing real errors."""
    pred_len = int(getattr(args, "pred_len", 96))
    label_len = int(getattr(args, "label_len", 0))
    dec_inp = _build_decoder_input(batch_y, pred_len, label_len)

    attempts = [
        (batch_x, batch_x_mark, dec_inp, batch_y_mark),
        (batch_x, batch_x_mark, None, None),
        (batch_x,),
    ]
    type_errors: List[str] = []
    for call_args in attempts:
        try:
            return model(*call_args)
        except TypeError as exc:
            type_errors.append(str(exc))
    raise TypeError(
        "Could not call the model with supported signatures. Errors: " + " | ".join(type_errors)
    )


def _capture_forward_geometry(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: Optional[torch.Tensor],
    batch_y_mark: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    splat_encoder = model.splatting_residual
    gaussian_module = splat_encoder.gaussian_splatting
    generator = gaussian_module.generator
    cas_gating = splat_encoder.cas_gating

    capture: Dict[str, Any] = {}

    def generator_hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("Gaussian generator hook did not receive a tensor input.")
        capture["generator_input"] = inputs[0].detach()
        capture["generator_output"] = _detach_tree(output)

    def gating_hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            capture["gating_input"] = inputs[0].detach()
        capture["gating_output"] = _detach_tree(output)

    def gaussian_hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        capture["gaussian_output"] = _detach_tree(output)

    handles = [
        generator.register_forward_hook(generator_hook),
        cas_gating.register_forward_hook(gating_hook),
        gaussian_module.register_forward_hook(gaussian_hook),
    ]
    try:
        model_output = _run_full_forward(
            model,
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            args,
        )
        capture["model_output"] = _detach_tree(model_output)
        capture["module_geometry_cache"] = _detach_tree(_geometry_from_module_cache(gaussian_module))
    finally:
        for handle in handles:
            handle.remove()

    if "generator_input" not in capture or "generator_output" not in capture:
        raise RuntimeError(
            "The real model forward did not invoke gaussian_splatting.generator. "
            "Check that the loaded checkpoint/config selects the intended GeoJet-TS pathway."
        )
    if "gating_output" not in capture:
        raise RuntimeError(
            "The real model forward did not invoke cas_gating. Check density_mode/ablation configuration."
        )
    return capture


def _decode_gating_output(output: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, Mapping):
        lower = {str(key).lower(): key for key in output.keys()}
        gate_key = next(
            (lower[name] for name in ("gate_effective", "gate", "hard_gate", "gates") if name in lower),
            None,
        )
        score_key = next(
            (lower[name] for name in ("pi_i", "pi", "retention_score", "probability") if name in lower),
            None,
        )
        if gate_key is None:
            raise RuntimeError(f"Could not identify gate tensor in router output keys: {list(output.keys())}")
        gate = output[gate_key]
        score = output[score_key] if score_key is not None else None
    elif isinstance(output, (tuple, list)) and len(output) >= 1:
        gate = output[0]
        score = output[1] if len(output) >= 2 and torch.is_tensor(output[1]) else None
    elif torch.is_tensor(output):
        gate = output
        score = None
    else:
        raise RuntimeError(f"Unsupported router output type: {type(output)!r}")

    if not torch.is_tensor(gate):
        raise RuntimeError("Router forward gate is not a tensor.")
    return gate.detach(), score.detach() if torch.is_tensor(score) else None


def _decode_geometry(
    capture: Mapping[str, Any],
    gaussian_module: torch.nn.Module,
    expected_rows: int,
    num_gaussians: int,
    require_forward_geometry: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Prefer constrained geometry exposed by the real forward; otherwise fallback."""
    candidates = [
        capture.get("module_geometry_cache"),
        _find_mapping_with_geometry(capture.get("gaussian_output")),
        _find_mapping_with_geometry(capture.get("model_output")),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            normalized = _normalize_geometry_mapping(candidate, expected_rows, num_gaussians)
            if normalized is not None:
                return normalized[0], normalized[1], normalized[2], "forward_exposed_geometry"

    if require_forward_geometry:
        raise RuntimeError(
            "The model forward did not expose constrained mu/sigma/alpha. Add a dictionary "
            "such as gaussian_splatting.last_geometry = {'mu': mu, 'sigma': sigma, "
            "'alpha': alpha} inside the real forward, or disable --require_forward_geometry."
        )

    raw_output = capture["generator_output"]
    if not torch.is_tensor(raw_output):
        raise RuntimeError(
            "Generator output is not a tensor and no forward-exposed geometry was found."
        )
    raw = raw_output.reshape(expected_rows, num_gaussians, -1)
    if raw.shape[-1] < 1:
        raise RuntimeError(f"Generator output has insufficient per-primitive fields: {raw.shape}")

    eps_sigma = float(getattr(gaussian_module, "eps_sigma", 1e-5))
    mu = torch.sigmoid(raw[:, :, 0])
    if hasattr(gaussian_module, "raw_sigma"):
        sigma_shared = F.softplus(gaussian_module.raw_sigma) + eps_sigma
        sigma = sigma_shared.unsqueeze(0).expand(expected_rows, -1)
        alpha = torch.ones_like(mu)
    else:
        sigma = F.softplus(raw[:, :, 1]) + eps_sigma
        alpha = torch.sigmoid(raw[:, :, 2])
    warnings.warn(
        "The model did not expose constrained geometry; using fallback parameterization. "
        "Verify this matches the actual model forward.",
        RuntimeWarning,
    )
    return mu, sigma, alpha, "fallback_generator_parameterization"


def _fallback_complexity(x_flat: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Replicate-padded moving average followed by absolute second variation."""
    if patch_len < 1:
        raise ValueError(f"patch_len must be positive, got {patch_len}")
    x = x_flat.unsqueeze(1)
    left = patch_len // 2
    right = patch_len - 1 - left
    x_pad = F.pad(x, (left, right), mode="replicate")
    x_low = F.avg_pool1d(x_pad, kernel_size=patch_len, stride=1)
    if x_low.shape[-1] != x.shape[-1]:
        raise RuntimeError(f"Unexpected smoothing length: {x_low.shape[-1]} vs {x.shape[-1]}")
    diff2 = x_low[..., 2:] - 2.0 * x_low[..., 1:-1] + x_low[..., :-2]
    return torch.sum(torch.abs(diff2), dim=-1).squeeze(1)


def _extract_complexity(
    cas_gating: torch.nn.Module,
    x_flat: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    for attr in ("last_complexity", "last_kappa", "complexity_cache", "_last_complexity"):
        value = getattr(cas_gating, attr, None)
        if torch.is_tensor(value) and value.numel() == x_flat.shape[0]:
            return value.detach().reshape(-1), f"router_cache:{attr}"

    for method_name in ("compute_complexity", "complexity_statistic", "get_complexity"):
        method = getattr(cas_gating, method_name, None)
        if callable(method):
            try:
                value = method(x_flat)
            except TypeError:
                continue
            if torch.is_tensor(value) and value.numel() == x_flat.shape[0]:
                return value.detach().reshape(-1), f"router_method:{method_name}"

    patch_len = int(getattr(cas_gating, "patch_len", 16) or 16)
    return _fallback_complexity(x_flat, patch_len), "fallback_replicate_padded_curvature"


# -----------------------------------------------------------------------------
# Extraction and sample selection
# -----------------------------------------------------------------------------

def _to_device_optional(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value.float().to(device)
    return None


def _iter_test_batches(test_loader: Iterable[Any]) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Any, Any]]:
    for batch in test_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError("Expected test loader batches with at least batch_x and batch_y.")
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x_mark = batch[2] if len(batch) > 2 else None
        batch_y_mark = batch[3] if len(batch) > 3 else None
        yield batch_x, batch_y, batch_x_mark, batch_y_mark


@torch.no_grad()
def extract_dataset_geometry(
    dataset_name: str,
    pred_len: int,
    seq_len: int,
    root_path: str,
    checkpoints_dir: str,
    device: torch.device,
    max_windows: int,
    channel_index: int,
    explicit_config: Optional[str],
    require_saved_config: bool,
    allow_non_strict: bool,
    require_forward_geometry: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n[Extract] Processing dataset={dataset_name}, pred_len={pred_len}...")
    checkpoint_path = find_checkpoint_path(checkpoints_dir, dataset_name, pred_len)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Checkpoint for {dataset_name} H={pred_len} was not found under {checkpoints_dir}."
        )
    print(f"  -> Checkpoint: {checkpoint_path}")

    args, config_source = build_args(
        dataset_name=dataset_name,
        pred_len=pred_len,
        seq_len=seq_len,
        root_path=root_path,
        checkpoints_dir=checkpoints_dir,
        checkpoint_path=checkpoint_path,
        explicit_config=explicit_config,
        require_saved_config=require_saved_config,
    )
    print(f"  -> Config source: {config_source}")

    model = Model(args).to(device)
    load_info = load_checkpoint(model, checkpoint_path, device, allow_non_strict)
    model.eval()

    _, test_loader = data_provider(args, flag="test")
    splat_encoder = model.splatting_residual
    gaussian_module = splat_encoder.gaussian_splatting
    cas_gating = splat_encoder.cas_gating
    num_gaussians = int(gaussian_module.num_gaussians)

    records: List[Dict[str, Any]] = []
    processed_windows = 0
    geometry_sources: set[str] = set()
    complexity_sources: set[str] = set()

    for batch_x_cpu, batch_y_cpu, batch_x_mark_cpu, batch_y_mark_cpu in _iter_test_batches(test_loader):
        if max_windows >= 0 and processed_windows >= max_windows:
            break

        remaining = None if max_windows < 0 else max_windows - processed_windows
        if remaining is not None and batch_x_cpu.shape[0] > remaining:
            batch_x_cpu = batch_x_cpu[:remaining]
            batch_y_cpu = batch_y_cpu[:remaining]
            if torch.is_tensor(batch_x_mark_cpu):
                batch_x_mark_cpu = batch_x_mark_cpu[:remaining]
            if torch.is_tensor(batch_y_mark_cpu):
                batch_y_mark_cpu = batch_y_mark_cpu[:remaining]

        batch_x = batch_x_cpu.float().to(device)
        batch_y = batch_y_cpu.float().to(device)
        batch_x_mark = _to_device_optional(batch_x_mark_cpu, device)
        batch_y_mark = _to_device_optional(batch_y_mark_cpu, device)
        batch_size, length, channels = batch_x.shape

        capture = _capture_forward_geometry(
            model,
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            args,
        )

        x_flat = capture["generator_input"]
        if x_flat.ndim != 2 or x_flat.shape != (batch_size * channels, length):
            raise RuntimeError(
                "Captured generator input does not match channel-independent B*C ordering: "
                f"captured={tuple(x_flat.shape)}, expected={(batch_size * channels, length)}"
            )

        gate_effective, retention_score = _decode_gating_output(capture["gating_output"])
        gate_effective = gate_effective.reshape(batch_size * channels, num_gaussians)
        # Straight-through gates use their hard value in the forward pass.
        gate_hard = (gate_effective > 0.5).to(dtype=x_flat.dtype)
        active_count = gate_hard.sum(dim=-1)

        mu, sigma, alpha, geometry_source = _decode_geometry(
            capture,
            gaussian_module,
            expected_rows=batch_size * channels,
            num_gaussians=num_gaussians,
            require_forward_geometry=require_forward_geometry,
        )
        geometry_sources.add(geometry_source)

        complexity, complexity_source = _extract_complexity(cas_gating, x_flat)
        complexity_sources.add(complexity_source)
        effective_alpha = alpha * gate_hard

        x_model_np = x_flat.detach().cpu().numpy().reshape(batch_size, channels, length)
        x_raw_np = batch_x.detach().cpu().numpy()
        complexity_np = complexity.detach().cpu().numpy().reshape(batch_size, channels)
        mu_np = mu.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        sigma_np = sigma.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        alpha_np = alpha.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        gate_effective_np = gate_effective.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        gate_hard_np = gate_hard.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        effective_alpha_np = effective_alpha.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
        active_count_np = active_count.detach().cpu().numpy().reshape(batch_size, channels)
        retention_np = (
            retention_score.detach().cpu().numpy().reshape(batch_size, channels, num_gaussians)
            if retention_score is not None
            else None
        )

        selected_channels: Sequence[int]
        if channel_index >= 0:
            if channel_index >= channels:
                raise IndexError(
                    f"Requested channel_index={channel_index}, but {dataset_name} has C={channels}."
                )
            selected_channels = (channel_index,)
        else:
            selected_channels = tuple(range(channels))

        for batch_row in range(batch_size):
            window_index = processed_windows + batch_row
            for channel in selected_channels:
                record: Dict[str, Any] = {
                    "dataset": dataset_name,
                    "horizon": pred_len,
                    "window_index": int(window_index),
                    "channel_index": int(channel),
                    "input_model": x_model_np[batch_row, channel].copy(),
                    # Compatibility alias used by the old plotting script.
                    "input_revin": x_model_np[batch_row, channel].copy(),
                    "input_raw": x_raw_np[batch_row, :, channel].copy(),
                    "complexity_kappa": float(complexity_np[batch_row, channel]),
                    "mu": mu_np[batch_row, channel].copy(),
                    "sigma": sigma_np[batch_row, channel].copy(),
                    "alpha": alpha_np[batch_row, channel].copy(),
                    "gate_effective": gate_effective_np[batch_row, channel].copy(),
                    "hard_gate": gate_hard_np[batch_row, channel].copy(),
                    "effective_alpha": effective_alpha_np[batch_row, channel].copy(),
                    "active_count": int(active_count_np[batch_row, channel]),
                    "geometry_source": geometry_source,
                    "complexity_source": complexity_source,
                }
                if retention_np is not None:
                    record["retention_score"] = retention_np[batch_row, channel].copy()
                records.append(record)

        processed_windows += batch_size

    metadata = {
        "dataset": dataset_name,
        "horizon": pred_len,
        "checkpoint": str(checkpoint_path),
        "config_source": config_source,
        "checkpoint_load": load_info,
        "geometry_sources": sorted(geometry_sources),
        "complexity_sources": sorted(complexity_sources),
        "num_gaussians": num_gaussians,
        "k_base": int(getattr(args, "k_base", -1)),
        "gate_tau": float(getattr(cas_gating, "tau", float("nan"))),
        "patch_len": int(getattr(cas_gating, "patch_len", getattr(args, "patch_len", -1)) or -1),
        "selected_channel": channel_index,
        "processed_windows": processed_windows,
        "total_sequences": len(records),
    }
    print(f"  -> Extracted {len(records)} sequence records from {processed_windows} windows.")
    print(f"  -> Geometry source(s): {metadata['geometry_sources']}")
    return records, metadata


def select_representative_samples(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if len(records) < 3:
        raise ValueError("At least three records are required for percentile selection.")
    sorted_records = sorted(records, key=lambda record: record["complexity_kappa"])
    count = len(sorted_records)

    def nearest_percentile(percentile: float) -> Dict[str, Any]:
        index = int(round(percentile * (count - 1)))
        return sorted_records[index]

    return {
        "low": nearest_percentile(0.10),
        "mid": nearest_percentile(0.50),
        "high": nearest_percentile(0.90),
    }


def save_diagnostic_plot(sample: Mapping[str, Any], save_path: Path) -> None:
    """Generate a compact support-strip diagnostic; final paper plotting is separate."""
    x_model = np.asarray(sample["input_model"])
    length = x_model.shape[0]
    time = np.arange(length)
    mu = np.asarray(sample["mu"])
    sigma = np.asarray(sample["sigma"])
    effective_alpha = np.asarray(sample["effective_alpha"])
    hard_gate = np.asarray(sample["hard_gate"])
    active_indices = np.flatnonzero(hard_gate > 0.5)

    mu_idx = mu * (length - 1)
    sigma_idx = sigma * (length - 1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, max(1.0, 0.30 * len(active_indices))]},
    )
    axes[0].plot(time, x_model, color="black", linewidth=1.2)
    axes[0].set_ylabel("Model input")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_title(
        f"{sample['dataset']} H={sample['horizon']} | window={sample['window_index']} "
        f"channel={sample['channel_index']} | kappa={sample['complexity_kappa']:.3f} "
        f"| K(x)={sample['active_count']}"
    )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(mu), 1)))
    for row, primitive in enumerate(active_indices, start=1):
        center = float(mu_idx[primitive])
        sigma_samples = float(sigma_idx[primitive])
        left_raw = center - 2.0 * sigma_samples
        right_raw = center + 2.0 * sigma_samples
        left = max(0.0, left_raw)
        right = min(length - 1.0, right_raw)
        saliency = float(effective_alpha[primitive])
        color = colors[primitive % len(colors)]

        axes[1].hlines(row, left, right, color=color, linewidth=2.2, alpha=0.35 + 0.65 * saliency)
        axes[1].scatter(
            [center],
            [row],
            color=color,
            s=35.0 + 130.0 * saliency,
            alpha=0.35 + 0.65 * saliency,
            zorder=4,
        )
        axes[1].text(
            length - 2,
            row,
            f"k={primitive}  mu={center:.1f}  sigma={sigma_samples:.1f}  a~={saliency:.2f}",
            ha="right",
            va="center",
            fontsize=8,
        )
        if left_raw < 0:
            axes[1].plot([0], [row], marker="<", color=color, markersize=5)
        if right_raw > length - 1:
            axes[1].plot([length - 1], [row], marker=">", color=color, markersize=5)

    axes[1].set_xlabel("Input timestep")
    axes[1].set_ylabel("Active primitive")
    axes[1].set_yticks(np.arange(1, len(active_indices) + 1))
    axes[1].set_yticklabels([f"k={index}" for index in active_indices])
    axes[1].set_ylim(0.4, max(1.6, len(active_indices) + 0.6))
    axes[1].set_xlim(0, length - 1)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _object_array(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray(list(records), dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GeoJet-TS geometry for Experiment 5.4 A1")
    parser.add_argument("--datasets", default="ETTh1,ETTm1,weather,electricity")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--root_path", default="./data")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--config", default=None, help="Explicit JSON config; use with one dataset")
    parser.add_argument("--require_saved_config", action="store_true")
    parser.add_argument("--allow_non_strict", action="store_true")
    parser.add_argument(
        "--require_forward_geometry",
        action="store_true",
        help="Fail unless the real model forward exposes constrained mu/sigma/alpha.",
    )
    parser.add_argument("--output_dir", default="./visualization/outputs/geometry_a1")
    parser.add_argument(
        "--max_windows",
        type=int,
        default=-1,
        help="Maximum test windows per dataset; -1 uses the complete test split.",
    )
    parser.add_argument(
        "--channel_index",
        type=int,
        default=0,
        help="Fixed channel used for clean percentile selection; -1 includes all channels.",
    )
    parsed = parser.parse_args()

    datasets = [name.strip() for name in parsed.datasets.split(",") if name.strip()]
    if parsed.config and len(datasets) != 1:
        parser.error("--config may only be used when exactly one dataset is requested.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_root = Path(parsed.output_dir)
    raw_dir = output_root / "raw"
    diagnostic_dir = output_root / "diagnostics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "pred_len": parsed.pred_len,
        "seq_len": parsed.seq_len,
        "max_windows": parsed.max_windows,
        "channel_index": parsed.channel_index,
        "datasets": {},
    }

    for dataset_name in datasets:
        try:
            records, metadata = extract_dataset_geometry(
                dataset_name=dataset_name,
                pred_len=parsed.pred_len,
                seq_len=parsed.seq_len,
                root_path=parsed.root_path,
                checkpoints_dir=parsed.checkpoints,
                device=device,
                max_windows=parsed.max_windows,
                channel_index=parsed.channel_index,
                explicit_config=parsed.config,
                require_saved_config=parsed.require_saved_config,
                allow_non_strict=parsed.allow_non_strict,
                require_forward_geometry=parsed.require_forward_geometry,
            )
        except Exception as exc:
            print(f"[Error] {dataset_name}: {exc}")
            manifest["datasets"][dataset_name] = {"error": str(exc)}
            continue

        selected = select_representative_samples(records)
        all_path = raw_dir / f"{dataset_name}_H{parsed.pred_len}_all_samples.npz"
        selected_path = raw_dir / f"{dataset_name}_H{parsed.pred_len}_selected_samples.npz"
        np.savez_compressed(all_path, records=_object_array(records), metadata=np.asarray(metadata, dtype=object))
        np.savez_compressed(
            selected_path,
            low=np.asarray(selected["low"], dtype=object),
            mid=np.asarray(selected["mid"], dtype=object),
            high=np.asarray(selected["high"], dtype=object),
            metadata=np.asarray(metadata, dtype=object),
        )

        for level, sample in selected.items():
            save_diagnostic_plot(
                sample,
                diagnostic_dir / f"{dataset_name}_H{parsed.pred_len}_{level}.png",
            )

        metadata["all_samples_npz"] = str(all_path)
        metadata["selected_samples_npz"] = str(selected_path)
        metadata["selected_examples"] = {
            level: {
                "window_index": int(sample["window_index"]),
                "channel_index": int(sample["channel_index"]),
                "complexity_kappa": float(sample["complexity_kappa"]),
                "active_count": int(sample["active_count"]),
            }
            for level, sample in selected.items()
        }
        manifest["datasets"][dataset_name] = metadata

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"\n[Success] Geometry extraction outputs: {output_root}")


if __name__ == "__main__":
    main()
