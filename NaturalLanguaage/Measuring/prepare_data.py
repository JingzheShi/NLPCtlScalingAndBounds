# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset  # huggingface datasets
from argparse import ArgumentParser
import torch
from datasets import load_dataset, DatasetDict

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 96

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

CACHE_DIR = '/scratch/project_xxxxxxxxxxx/HUGGINGFACE_CACHEDIR'

model_path = "/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache/Llama-3.1-8B"

# Load tokenizer and model with device_map='auto' to shard across GPUs
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

def encode_length_filter(example):
    encoded = tokenizer.encode(example['text'], add_special_tokens=False)
    filter_indicator = len(encoded) > 3000
    return {"filter_indicator": filter_indicator}

def process(example):
    ids = tokenizer.encode(example['text'], add_special_tokens=False)
    ids.append(tokenizer.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--percent', type=float, default=10)
    parser.add_argument('--start_from', type=float, default=0.0)
    parser.add_argument('--save_path', type=str, default='./filtered_dataset', help='Path to save the filtered dataset')
    args = parser.parse_args()
    PERCENT = args.percent
    START_FROM = args.start_from

    # Load the dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, cache_dir=CACHE_DIR, trust_remote_code=True)

    # Apply the encode_length_filter function
    dataset["train"] = dataset["train"].map(encode_length_filter, num_proc=num_proc)

    # Filter out the short contexts
    dataset["train"] = dataset["train"].filter(lambda x: x['filter_indicator'] == True)
    # Remove the filter indicator
    dataset["train"] = dataset["train"].remove_columns('filter_indicator')

    # Create a validation split
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=114, shuffle=False)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to val

    # Select a subset of the dataset
    start_from = int((START_FROM) / 100.0 * len(split_dataset['train']))
    num_train_samples = int(PERCENT / 100.0 * len(split_dataset['train']))
    split_dataset['train'] = split_dataset['train'].select(range(start_from, start_from + num_train_samples))

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        batch_size=30,
    )

    # Determine the appropriate dtype based on the tokenizer's vocabulary size
    max_token_id = max(tokenizer.get_vocab().values())
    print(f"max token id is {max_token_id}")

    # Save the tokenized data
    
    print("Start to save dataset.")
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'val':split_dataset['val']
    })
    
    dataset_dict.save_to_disk(args.save_path)
    
    print("Dataset preprocessing and saving completed")
    
    
    
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset['len'], dtype=np.uint64)
    #     filename = os.path.join(os.path.dirname(__file__), f'LONGCONTEXTSUBSET_{split}_P{PERCENT}_SFrom{START_FROM}.bin')
    #     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    #     total_batches = 100

    #     idx = 0
    #     for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
    #         # Batch together samples for faster write
    #         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    #         arr_batch = np.concatenate(batch['ids'])
    #         # Write into mmap
    #         arr[idx: idx + len(arr_batch)] = arr_batch
    #         idx += len(arr_batch)
    #     arr.flush()

    # To read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16 or np.uint32, mode='r')
