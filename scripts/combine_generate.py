import json
import pandas as pd
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--numgpu", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()

    for j in range(args.pairs):
        results = []
        if args.gpu_ids is not None:
            gpus = args.gpu_ids.strip("()").split(',')
        else:
            gpus = range(args.numgpu)

        for i in gpus:
            file_path = f"{args.output_dir}/responses_{i}_{j}.json"
            print(f'Reading from {file_path}')
            with open(file_path) as f:
                gen = json.load(f)
                results += gen

        output_path = f"{args.output_dir}/responses_{j}.json"
        print(f'Saved to {output_path}')
        with open(output_path, "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
