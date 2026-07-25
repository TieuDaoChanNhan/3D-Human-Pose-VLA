import torch
import numpy as np
from transformers import AutoTokenizer

# 1. Specify the model and Megatron binary file paths
model_path = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/vla_25b_test/hf/iter_0006000"
binary_dataset_path = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/alexandria/tokenized_output_project/vla_25b/data_shard_00000.bin"  # fill in the actual .bin file used at training time

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("💾 Loading Megatron binary training shards...")
# Megatron typically stores tokens as uint16 or int32 depending on the preprocess config
data = np.fromfile(binary_dataset_path, dtype=np.uint16)

# Sample the first 500 tokens from the actual training set
sample_token_ids = data[:500].tolist()

print("\n🔍 DETOKENIZING RAW TRAINING DATA TO CHECK FOR FRAGMENTATION...")
for token_id in sample_token_ids:
    decoded_piece = tokenizer.decode([token_id])
    # If output shows individual characters '<', 'seed', '2', '_' → tokenizer was messed up at train time
    # If output shows the whole token '<seed2_6750>' → the model has emergent ability
    print(f"ID: {token_id:<6} ──► Raw Text Fragment: [{decoded_piece}]")
