# Memory-Cached HPO Zero-Disk Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement in-memory checkpoint caching for HPO (Hyperparameter Optimization) trials using a new `--no_save_checkpoint` flag, bypassing physical disk I/O to protect SSD lifespans and avoid file manager hangs.

**Architecture:** Under the `--no_save_checkpoint` flag, `EarlyStopping` caches `model.state_dict()` in system RAM (copied to CPU to prevent VRAM accumulation) instead of calling `torch.save()`. The experiment coordinators (`Exp_Long_Term_Forecast` and `Exp_Short_Term_Forecast`) skip directory creation and reload model state from RAM at training termination. All HPO tuning runner scripts will inject this flag by default.

**Tech Stack:** PyTorch, Python Standard unittest, Optuna.

---

### Task 1: Write EarlyStopping Memory Cache Test and Verify Failure

**Files:**
- Create: `tests/test_no_save_checkpoint.py`

- [ ] **Step 1: Create the test file with failing test cases**
Create the new test file to verify the memory-cached saving logic in `EarlyStopping`.

Write this code to `tests/test_no_save_checkpoint.py`:
```python
import unittest
import os
import shutil
import torch
import torch.nn as nn
from utils.tools import EarlyStopping

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

class TestNoSaveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_checkpoints_temp"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.model = SimpleModel()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_checkpoint_to_disk_when_false(self):
        # Default behavior: no_save_checkpoint=False (writes to disk)
        early_stopping = EarlyStopping(patience=5, verbose=False)
        self.assertFalse(hasattr(early_stopping, "no_save_checkpoint") and early_stopping.no_save_checkpoint)
        
        # Trigger checkpoint save
        early_stopping(val_loss=1.0, model=self.model, path=self.test_dir)
        
        # Verify file is created on disk
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pth")
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_save_checkpoint_to_ram_when_true(self):
        # New behavior: no_save_checkpoint=True (caches in RAM)
        early_stopping = EarlyStopping(patience=5, verbose=False, no_save_checkpoint=True)
        self.assertTrue(early_stopping.no_save_checkpoint)
        self.assertIsNone(early_stopping.best_model_state)
        
        # Trigger checkpoint save
        early_stopping(val_loss=1.0, model=self.model, path=self.test_dir)
        
        # Verify no folder or file is created on disk
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pth")
        self.assertFalse(os.path.exists(checkpoint_path))
        
        # Verify state is saved in CPU RAM
        self.assertIsNotNone(early_stopping.best_model_state)
        # Check that tensor location is CPU
        for name, tensor in early_stopping.best_model_state.items():
            self.assertEqual(tensor.device, torch.device("cpu"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**
Run the test command inside the `demo` conda environment on Windows.

Run command:
```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python -m unittest tests/test_no_save_checkpoint.py"
```

Expected Output:
```
TypeError: __init__() got an unexpected keyword argument 'no_save_checkpoint'
FAILED (errors=1)
```

- [ ] **Step 3: Commit initial test setup**
```bash
git add tests/test_no_save_checkpoint.py
git commit -m "test: add TestNoSaveCheckpoint unit test to verify in-memory early stopping"
```

---

### Task 2: Implement EarlyStopping Memory-Caching Logic

**Files:**
- Modify: `utils/tools.py`

- [ ] **Step 1: Modify EarlyStopping __init__ and save_checkpoint**
Edit `utils/tools.py` to add `no_save_checkpoint` support.

Target: [EarlyStopping L81-112](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/utils/tools.py#L81-L112)
Replacement Content:
```python
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, no_save_checkpoint=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.no_save_checkpoint = no_save_checkpoint
        self.best_model_state = None

    def __call__(self, val_loss, model, path):
        # self.save_checkpoint(val_loss, model, path)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.no_save_checkpoint:
            # Store in CPU memory to avoid accumulated GPU VRAM memory leaks
            import copy
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
```

- [ ] **Step 2: Run unit test to verify it passes**
Run command:
```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python -m unittest tests/test_no_save_checkpoint.py"
```

Expected Output:
```
..
----------------------------------------------------------------------
Ran 2 tests in X.XXs

OK
```

- [ ] **Step 3: Commit EarlyStopping memory modifications**
```bash
git add utils/tools.py
git commit -m "feat: implement in-memory early stopping state dict caching"
```

---

### Task 3: Support `--no_save_checkpoint` Commandline Argument

**Files:**
- Modify: `run.py`

- [ ] **Step 1: Add parameter to argument parser**
Target: [run.py L79-81](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/run.py#L79-L81)
Replacement Content:
```python
    # optimization
    parser.add_argument("--no_save_checkpoint", action="store_true", default=False, help="Cache weights in RAM and skip writing checkpoints to disk.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
```

- [ ] **Step 2: Verification of argument parsed successfully**
Run command:
```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python run.py --help"
```
Verify output prints `--no_save_checkpoint` under optimization options.

- [ ] **Step 3: Commit argument parser additions**
```bash
git add run.py
git commit -m "feat: expose --no_save_checkpoint option in run.py parser"
```

---

### Task 4: Integrate Memory Caching in Long/Short Term Forecasters

**Files:**
- Modify: `exp/exp_long_term_forecasting.py`
- Modify: `exp/exp_short_term_forecasting.py`
- Modify: `tests/test_no_save_checkpoint.py`

- [ ] **Step 1: Add integration test case for Exp Controller training**
Add `test_exp_train_memory_caching` to `tests/test_no_save_checkpoint.py` using Mocks.
Add this case inside class `TestNoSaveCheckpoint` in `tests/test_no_save_checkpoint.py`:
```python
    def test_exp_train_memory_caching(self):
        from unittest.mock import MagicMock, patch
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        
        args = MagicMock()
        args.checkpoints = self.test_dir
        args.patience = 3
        args.train_epochs = 1
        args.no_save_checkpoint = True
        args.use_gpu = False
        args.use_multi_gpu = False
        args.model = "SplatTS"
        
        with patch('exp.exp_long_term_forecasting.Exp_Long_Term_Forecast._build_model') as mock_build_model, \
             patch('exp.exp_long_term_forecasting.Exp_Long_Term_Forecast._get_data') as mock_get_data, \
             patch('exp.exp_long_term_forecasting.adjust_learning_rate') as mock_adj_lr:
            
            # Setup mock model and data loader
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {"param": torch.tensor([1.0])}
            mock_build_model.return_value = mock_model
            
            mock_loader = MagicMock()
            mock_loader.__len__.return_value = 1
            mock_loader.__iter__.return_value = iter([(torch.zeros(1, 10, 1), torch.zeros(1, 10, 1), None, None)])
            mock_get_data.return_value = (None, mock_loader)
            
            exp = Exp_Long_Term_Forecast(args)
            exp.device = torch.device("cpu")
            exp._select_optimizer = MagicMock()
            exp._select_criterion = MagicMock()
            exp.vali = MagicMock(return_value=0.5)
            
            # Run train
            exp.train("test_setting")
            
            # Verify no checkpoints directory was ever created
            self.assertFalse(os.path.exists(self.test_dir))
```

- [ ] **Step 2: Modify `exp_long_term_forecasting.py`**
Target: [exp_long_term_forecasting.py L106-112](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/exp/exp_long_term_forecasting.py#L106-L112)
Replacement:
```python
        path = os.path.join(self.args.checkpoints, setting)
        if not getattr(self.args, "no_save_checkpoint", False):
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True, 
            no_save_checkpoint=getattr(self.args, "no_save_checkpoint", False)
        )
```

Target: [exp_long_term_forecasting.py L288-291](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/exp/exp_long_term_forecasting.py#L288-L291)
Replacement:
```python
        if getattr(self.args, "no_save_checkpoint", False):
            if early_stopping.best_model_state is not None:
                self.model.load_state_dict(early_stopping.best_model_state)
            else:
                print("Warning: No best model state found in memory.")
        else:
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model
```

- [ ] **Step 3: Modify `exp_short_term_forecasting.py`**
Target: [exp_short_term_forecasting.py L57-64](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/exp/exp_short_term_forecasting.py#L57-L64)
Replacement:
```python
        path = os.path.join(self.args.checkpoints, setting)
        if not getattr(self.args, "no_save_checkpoint", False):
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True, 
            no_save_checkpoint=getattr(self.args, "no_save_checkpoint", False)
        )
```

Target: [exp_short_term_forecasting.py L127-130](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/exp/exp_short_term_forecasting.py#L127-L130)
Replacement:
```python
        if getattr(self.args, "no_save_checkpoint", False):
            if early_stopping.best_model_state is not None:
                self.model.load_state_dict(early_stopping.best_model_state)
            else:
                print("Warning: No best model state found in memory.")
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model
```

- [ ] **Step 4: Run unit and integration tests**
Run command:
```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python -m unittest tests/test_no_save_checkpoint.py"
```
Confirm all 3 tests pass successfully.

- [ ] **Step 5: Commit integration modifications**
```bash
git add exp/exp_long_term_forecasting.py exp/exp_short_term_forecasting.py tests/test_no_save_checkpoint.py
git commit -m "feat: integrate --no_save_checkpoint into long/short-term forecasters"
```

---

### Task 5: Enable `--no_save_checkpoint` in HPO Tuning Runner Scripts

**Files:**
- Modify: `tune_gate_etth1.py`
- Modify: `tune_gate_etth2.py`
- Modify: `tune_gate_ettm1.py`
- Modify: `tune_gate_ettm2.py`

- [ ] **Step 1: Update tune_gate_etth1.py**
Target: [tune_gate_etth1.py L256-260](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_etth1.py#L256-L260)
Replacement:
```python
    if gate_type == "adaptive_direction":
        cmd.extend(["--gate_lambda", str(params.get("gate_lambda", 0.0))])
        cmd.extend(["--gate_window_half", str(params.get("gate_window_half", 2))])
    cmd.extend(["--no_save_checkpoint"])
    cmd.extend(["--output_dir", output_dir])
    return cmd
```

- [ ] **Step 2: Update tune_gate_etth2.py**
Open [tune_gate_etth2.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_etth2.py), locate `build_command`, and inject `--no_save_checkpoint` at the end of the argument builders.

- [ ] **Step 3: Update tune_gate_ettm1.py**
Open [tune_gate_ettm1.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_ettm1.py), locate `build_command`, and inject `--no_save_checkpoint` at the end of the argument builders.

- [ ] **Step 4: Update tune_gate_ettm2.py**
Open [tune_gate_ettm2.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_ettm2.py), locate `build_command`, and inject `--no_save_checkpoint` at the end of the argument builders.

- [ ] **Step 5: Run HPO trial sanity verification**
Run HPO scripts manually for 1-2 trials to verify:
- It terminates normally.
- No files are written in checkpoints directory.
Run command:
```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python tune_gate_etth1.py --n_trials 2"
```

- [ ] **Step 6: Commit runner updates**
```bash
git add tune_gate_etth1.py tune_gate_etth2.py tune_gate_ettm1.py tune_gate_ettm2.py
git commit -m "feat: default enable --no_save_checkpoint flag for all HPO runners"
```
