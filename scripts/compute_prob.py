import numpy as np
from datasets import load_dataset, Dataset
import json
import argparse
import pandas as pd
import datasets
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--prompts", type=str, default="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1")
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=8)
    parser.add_argument("--org", type=str, default="UCLA-AGI")
    parser.add_argument("--gpu_ids", type=str, default=None)
    return parser.parse_args()

def from_ranks(args):
    num_gpu = args.num_gpu
    pairs = args.pairs
    data = load_dataset(args.prompts, split="train")
    print(f"Length of dataset: {len(data)}")

    scores = [0 for _ in range(len(data))]
    if args.gpu_ids is not None:
        gpus = args.gpu_ids.strip("()").split(',')
    else:
        gpus = range(args.num_gpu)

    for data_frac, idx in enumerate(gpus):
        locals = np.load(f"ranking/{args.output_dir}/{idx}_{data_frac}.npy")
        locals = list(locals)
        for lidx, sc in enumerate(locals):
            scores[data_frac * args.frac_len + lidx] = sc

    probs = []
    rm_scores = []
    for idx, score in enumerate(scores):
        prb = np.zeros((pairs, pairs))
        for i in range(pairs):
            for j in range(pairs):
                prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

    print("Saving probabilities...")
    with open(f"generated/{args.output_dir}/probabilities.json", "w") as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(pairs):
        with open(f"generated/{args.output_dir}/responses_{i}.json") as f:
            responses = json.load(f)
        fmt = [
            [
                {"content": data[j]["prompt"], "role": "user"},
                {"content": responses[j], "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f"generate_{i}"] = fmt

    df["probability"] = probs
    df["rm_scores"] = rm_scores
    df.to_parquet(f"generated/{args.output_dir}/train.parquet")


import numpy as np
import os
import pandas as pd
import datasets

def prepare_score(args):
    # Load dataset and convert to DataFrame
    train = datasets.load_dataset(f"generated/{args.output_dir}")
    train = pd.DataFrame(train['train'])

    # Calculate metrics and probabilities
    metrics = train['rm_scores'].apply(lambda x: np.array(x[-5:]))
    metrics_prob = train['probability'].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    # Reorganize the DataFrame for easy access
    train_ordered = train[['generate_0', 'generate_1', 'generate_2', 'generate_3', 'generate_4', 'probability']]

    # Determine chosen and rejected items based on maxmin indices
    chosen = [train_ordered.iloc[i, maxmin[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] for i in range(len(train_ordered))]

    # Calculate probabilities for chosen and rejected items
    chosen_probs = [train_ordered['probability'].iloc[i][maxmin[i][0]][maxmin[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    # Create a new DataFrame with the results
    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })

    # Determine output directory
    output_dir = '-'.join(args.output_dir.split('-')[1:])
    OUTPATH = f'synthetic_data_{output_dir}_score'
    os.makedirs(OUTPATH, exist_ok=True)

    # Save train and test datasets to parquet files
    train_new.to_parquet(f'{OUTPATH}/train.parquet', index=False)
    print(f"Saved file to {OUTPATH}/train.parquet")

    # Temporary solution to make the code run, cannot use for test/evaluation purpose
    test = train_new.sample(n=500)
    test.to_parquet(f'{OUTPATH}/test.parquet', index=False)
    print(f"Saved file to {OUTPATH}/test.parquet")

    return OUTPATH

def push_dataset(file_dir, org):
    data = Dataset.from_parquet(f"{file_dir}/train.parquet")
    try:
        test = Dataset.from_parquet(f"{file_dir}/test.parquet")
    except:
        train = pd.read_parquet(f"{file_dir}/train.parquet")
        # Temporary solution to make the code run, cannot use for test/evaluation purpose
        test = train.sample(n=500)
        test.to_parquet(f"{file_dir}/test.parquet", index=False)
        test = Dataset.from_parquet(f"{file_dir}/test.parquet")
    data.push_to_hub(f"{org}/{file_dir}", split="train", private=True)
    test.push_to_hub(f"{org}/{file_dir}", split="test", private=True)



if __name__ == "__main__":
    args = parse_arguments()
    from_ranks(args)
    data = Dataset.from_parquet(f"generated/{args.output_dir}/train.parquet")
    data.push_to_hub(f"{args.org}/{args.output_dir}_generated", private=True)
    out_path = prepare_score(args)
    push_dataset(out_path, args.org)
