SEED = 114
import os
import json
import torch
import numpy as np
import random
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
TASK_LIST = json.load(open("task_definition.json"))
CURRENT_L_MAX = 100
num_task = len(TASK_LIST)
TRAIN_DATASET_SIZE = 10000000
VALID_DATASET_SIZE = 2000000
all_size = TRAIN_DATASET_SIZE + VALID_DATASET_SIZE
from tqdm import tqdm
def generate_unique_binary_tensor(N, L):
    """
    Generate a tensor of N unique binary sequences of length L.
    
    Args:
        N (int): Number of sequences (must be <= 2^L)
        L (int): Length of each sequence
    
    Returns:
        torch.Tensor: Tensor of shape (N, L) containing unique binary sequences
    """
    # Check if N is valid
    max_sequences = 2**L
    if N > max_sequences:
        raise ValueError(f"N ({N}) cannot be larger than 2^L ({max_sequences})")
    
    # Initialize set to keep track of seen sequences
    seen_sequences = set()
    result = torch.zeros((N, L), dtype=torch.int)
    
    for i in tqdm(range(N)):
        while True:
            # Generate a random binary sequence
            sequence = torch.randint(0, 2, (L,), dtype=torch.int)
            
            # Convert to tuple for hashability
            sequence_tuple = tuple(sequence.tolist())
            
            # Check if sequence is unique
            if sequence_tuple not in seen_sequences:
                seen_sequences.add(sequence_tuple)
                result[i] = sequence
                break
    
    return result

result = generate_unique_binary_tensor(all_size, CURRENT_L_MAX)
train_data = result[:TRAIN_DATASET_SIZE]
valid_data = result[TRAIN_DATASET_SIZE:]

def generate_answer_and_ctrl_bits(result, all_size, CURRENT_L_MAX):
    answers = torch.zeros((all_size), dtype=torch.int)
    ctrl_bits = torch.zeros((all_size, num_task), dtype=torch.int)
    for idx in tqdm(range(all_size)):
        task = TASK_LIST[idx%num_task]
        ctrl_bits[idx, idx%num_task] = 1
        l, a = task
        # print(l,a,CURRENT_L_MAX)
        if l <= CURRENT_L_MAX:
            data = result[idx]
            answer = data[a-1] ^ data[l-1]
            answers[idx] = answer
        else:
            answers[idx] = random.randint(0, 1)
    return answers, ctrl_bits
answers, ctrl_bits = generate_answer_and_ctrl_bits(result, all_size, CURRENT_L_MAX)
train_answers = answers[:TRAIN_DATASET_SIZE]
train_ctrl_bits = ctrl_bits[:TRAIN_DATASET_SIZE]
valid_answers = answers[TRAIN_DATASET_SIZE:]
valid_ctrl_bits = ctrl_bits[TRAIN_DATASET_SIZE:]

os.makedirs("data", exist_ok=True)
torch.save(dict(
    task_bits = result[:TRAIN_DATASET_SIZE],
    ctrl_bits = train_ctrl_bits,
    answers = train_answers,
), "data/train.pt")
torch.save(dict(
    task_bits = result[TRAIN_DATASET_SIZE:],
    ctrl_bits = valid_ctrl_bits,
    answers = valid_answers,
), "data/valid.pt")