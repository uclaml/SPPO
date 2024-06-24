from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='generated/iter1')
    parser.add_argument('--org', type=str, default='UCLA-AGI')
    parser.add_argument('--id', type=str, default="")
    return parser.parse_args()

def push(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.push_to_hub(f"{args.org}/{args.id}", private=True, max_shard_size='5GB')
    tokenizer.push_to_hub(f"{args.org}/{args.id}", private=True)

if __name__ == "__main__":
    args = parse_arguments()
    push(args)
