"""Run the comparison from a terminal; see README.md for setup."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare LSTM, Transformer and GNN train-delay predictions.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/latest")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--check", action="store_true", help="Prepare data and print split sizes without training.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.history) < 1:
        parser.error("epochs, batch-size and history must be positive")

    from train_delay.data import DelayData
    from train_delay.training import run_experiment

    data = DelayData(args.data_dir, history=args.history, batch_size=args.batch_size)
    print(data.source_stats)
    print(data.summary.to_string(index=False))
    if not args.check:
        run_experiment(data, args.output_dir, epochs=args.epochs)
        print(f"Saved this run to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
