import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--config', type=str, default='recipes/uclaml-sppo/config_full.yaml')
args=parser.parse_args()
# The path to your configuration file
file_path = args.config
# New dataset_mixer content you want to insert
new_dataset_mixer = f'dataset_mixer:\n  {args.dataset}: 1.0'

# Read the original content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Regular expression to match the dataset_mixer block and replace it
# Adjust the pattern if your structure might vary significantly
pattern = re.compile(r'dataset_mixer:\n\s*[^:]+:\s*\d+(\.\d+)?')

# Replace the matched pattern with the new dataset_mixer content
new_content = re.sub(pattern, new_dataset_mixer, content)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(new_content)

print("Dataset mixer updated successfully.")
