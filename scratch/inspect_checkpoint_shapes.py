import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu")
        print(path)
        for key, value in state.items():
            if (
                "head_srs" in key
                or "projection.weight" in key
                or "generator.4.weight" in key
                or "generator.3.weight" in key
                or "generator.2.weight" in key
            ):
                print(f"  {key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
