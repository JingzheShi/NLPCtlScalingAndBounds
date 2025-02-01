# evaluate_model.py

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import math
from tqdm import tqdm
import time
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on filtered dataset")
    parser.add_argument('--dataset_path', type=str, default='./filtered_dataset')
    parser.add_argument('--model_name', type=str, default="/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache/Llama-3.1-70B")
    parser.add_argument('--seq_lens', nargs='+', type=int, default=[1024])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iter',type=int,default=400)
    parser.add_argument('--dataset_column_name',type=str,default='text')
    parser.add_argument('--cache_chunk_file_name',type=str,default='chunks_Nstrict_and_to_example_startfrom50_toM50_biggerthan3000_2048sq.pt')
    args = parser.parse_args()

    CACHE_DIR = '/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache'
    
    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=True, device_map="auto")
    model.eval()

    print(f"Loading filtered dataset from {args.dataset_path}...")
    
    dataset = load_from_disk(args.dataset_path)['train']
    
    if True:    
        device = model.get_input_embeddings().weight.device
        
        num_examples = len(dataset)
        
        
        chunks = []
        chunk_to_example = []
        
        try:
            chunk_dict = torch.load(args.cache_chunk_file_name)
            chunks = chunk_dict['chunks']
            chunk_to_example = chunk_dict['chunk_to_example']
        except Exception as e:
            SEQ_LEN = 2048
            print(e)
            for example_idx, example in enumerate(tqdm(dataset, desc="Processing dataset")):
                text = example[args.dataset_column_name]
                tokens = tokenizer.encode(text)
                tokens = tokens[100:]
                tokens = tokens[:-50]
                total_tokens = len(tokens)

                for i in range(0, total_tokens - SEQ_LEN + 1, SEQ_LEN):
                    chunk = tokens[i:i+SEQ_LEN]
                    chunks.append(chunk)
                    chunk_to_example.append(example_idx)

                if total_tokens % SEQ_LEN > 0 and total_tokens > SEQ_LEN:
                    continue
                    chunk = tokens[-SEQ_LEN:]
                    chunks.append(chunk)
                    chunk_to_example.append(example_idx)

            print('save tensors')
            torch.save({
                'chunks': chunks,
                'chunk_to_example': chunk_to_example,
            }, args.cache_chunk_file_name)

        batch_size = args.batch_size
        num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size != 0 else 0)
        

        
        
        org_chunks = torch.tensor(chunks, dtype=torch.long, device=device)
        chunk_to_example = torch.tensor(chunk_to_example, device=device)
    for seq_len in args.seq_lens:
        
        chunks = org_chunks[:, -seq_len:,...]
        per_example_sum_CE = torch.zeros(num_examples, device=device)
        per_example_count = torch.zeros(num_examples, device=device)
        print(f"\nEvaluating for context length: {seq_len}")
        counter = 0
        for batch_idx in tqdm(range(num_batches), desc=f"Seq_len {seq_len}"):
            counter += 1
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            
            batch_chunks = chunks[start_idx:end_idx]
            batch_example_idxs = chunk_to_example[start_idx:end_idx]

            with torch.no_grad():
                outputs = model(batch_chunks)
                logits = outputs.logits
                shift_logits = logits[:, -2:-1, :].contiguous()
                shift_labels = batch_chunks[:, -1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                seq_losses = loss.view(-1)  # Flatten to 1D

                # Update sums and counts on GPU using scatter_add_
                per_example_sum_CE.scatter_add_(0, batch_example_idxs, seq_losses)
                per_example_count.scatter_add_(0, batch_example_idxs, torch.ones_like(seq_losses))

            if args.max_iter > 0:
                if counter > args.max_iter:
                    break

        # Move to CPU only at the end for final calculations
        mask = per_example_count > 0
        valid_perplexities = per_example_sum_CE[mask] / per_example_count[mask]
        valid_counts = per_example_count[mask]
        
        # Calculate final average
        if len(valid_perplexities) > 0:
            average_perplexity = (valid_perplexities * valid_counts).sum() / valid_counts.sum()
            print(f"Average Perplexity for seq_len {seq_len}: {average_perplexity.item()}")
        else:
            print(f"No valid perplexity values computed for seq_len {seq_len}.")

if __name__ == '__main__':
    main()

# 
# # evaluate_model.py

# import argparse
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_from_disk
# import math
# from tqdm import tqdm

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate model on filtered dataset")
#     parser.add_argument('--dataset_path', type=str, default='./filtered_dataset', help='Path to the filtered dataset')
#     parser.add_argument('--model_name', type=str, default="/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache/Llama-3.1-8B", help='Model name or path')
#     parser.add_argument('--seq_lens', nargs='+', type=int, default=[256], help='List of sequence lengths to evaluate')
    
#     args = parser.parse_args()

#     # Set the cache directory
#     CACHE_DIR = '/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache'

#     # Load the tokenizer and model with cache_dir
#     print(f"Loading model and tokenizer: {args.model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
#     model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=True, device_map="auto")
#     model.eval()

#     # Load the local dataset (no cache_dir needed since it's local)
#     print(f"Loading filtered dataset from {args.dataset_path}...")
#     dataset = load_from_disk(args.dataset_path)['train']

#     # Evaluation function
#     def evaluate_perplexity(text, seq_len):
#         # Tokenize the text starting from the 1,000th token
#         tokens = tokenizer.encode(text)
#         tokens = tokens[1100:]  # Start from the 1,000th token

#         total_tokens = len(tokens)
#         perplexities = []

#         # Process the tokens in chunks of seq_len
#         device = model.get_input_embeddings().weight.device
#         for i in range(0, total_tokens - seq_len + 1, seq_len):
#             input_ids = torch.tensor(tokens[i:i+seq_len]).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 outputs = model(input_ids, labels=input_ids)
#                 loss = outputs.loss.item()
#                 perplexity = math.exp(loss)
#                 perplexities.append(perplexity)

#         # Handle the remaining tokens if any
#         remainder = total_tokens % seq_len
#         if remainder > 0 and total_tokens > seq_len:
#             input_ids = torch.tensor(tokens[-seq_len:]).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 outputs = model(input_ids, labels=input_ids)
#                 loss = outputs.loss.item()
#                 perplexity = math.exp(loss)
#                 perplexities.append(perplexity)

#         # Calculate the average perplexity for the text
#         if perplexities:
#             return sum(perplexities) / len(perplexities)
#         else:
#             return None

#     # Evaluate the model for each specified seq_len
#     results = {}

#     for seq_len in args.seq_lens:
#         print(f"\nEvaluating for context length: {seq_len}")
#         perplexity_list = []

#         for example in tqdm(dataset, desc=f"Seq_len {seq_len}"):
#             text = example[args.dataset_column_name]
#             perplexity = evaluate_perplexity(text, seq_len)
#             if perplexity is not None:
#                 perplexity_list.append(perplexity)

#         # Calculate the average perplexity for the current seq_len
#         if perplexity_list:
#             average_perplexity = sum(perplexity_list) / len(perplexity_list)
#             results[seq_len] = average_perplexity
#             print(f"Average Perplexity for seq_len {seq_len}: {average_perplexity}")
#         else:
#             print(f"No valid perplexity values computed for seq_len {seq_len}.")
#             results[seq_len] = None

#     # Display the final results
#     print("\nFinal Results:")
#     for seq_len, avg_CE in results.items():
#         if avg_CE is not None:
#             print(f"Context Length {seq_len}: Average Perplexity = {avg_CE}")
#         else:
#             print(f"Context Length {seq_len}: No data to compute perplexity.")

# if __name__ == '__main__':
#     main()
