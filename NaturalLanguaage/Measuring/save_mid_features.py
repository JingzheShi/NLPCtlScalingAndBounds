import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import math
from tqdm import tqdm
import time
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on filtered dataset")
    parser.add_argument('--dataset_path', type=str, default='./filtered_dataset')
    parser.add_argument('--model_name', type=str, default="/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache/Llama-3.1-8B")
    parser.add_argument('--seq_lens', nargs='+', type=int, default=[32,384])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iter',type=int,default=400)
    args = parser.parse_args()

    CACHE_DIR = '/scratch/project_xxxxxxxxxxx/LLM_DID/huggingface_cache'
    SAVE_DIR = '/scratch/project_xxxxxxxxxxx/NLPContextScaling/llama/save_feature_dir_8B_new'
    
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        local_files_only=True, 
        device_map="auto",
        output_hidden_states=True  # Enable output of hidden states
    )
    model.eval()

    print(f"Loading filtered dataset from {args.dataset_path}...")
    
    dataset = load_from_disk(args.dataset_path)['train']
    
    if True:    
        device = model.get_input_embeddings().weight.device
        
        num_examples = len(dataset)
        
        chunks = []
        chunk_to_example = []
        
        try:
            chunk_dict = torch.load('chunks_Nstrict_and_to_example_startfrom50_toM50_biggerthan3000_2048sq.pt')
            chunks = chunk_dict['chunks']
            chunk_to_example = chunk_dict['chunk_to_example']
        except Exception as e:
            SEQ_LEN = 2048
            print(e)
            for example_idx, example in enumerate(tqdm(dataset, desc="Processing dataset")):
                text = example['text']
                tokens = tokenizer.encode(text)
                tokens = tokens[100:]
                tokens = tokens[:-50]
                total_tokens = len(tokens)

                for i in range(0, total_tokens - SEQ_LEN + 1, SEQ_LEN):
                    chunk = tokens[i:i+SEQ_LEN]
                    chunks.append(chunk)
                    chunk_to_example.append(example_idx)

                if total_tokens % SEQ_LEN > 0 and total_tokens > SEQ_LEN:
                    chunk = tokens[-SEQ_LEN:]
                    chunks.append(chunk)
                    chunk_to_example.append(example_idx)

            print('save tensors')
            torch.save({
                'chunks': chunks,
                'chunk_to_example': chunk_to_example,
            }, 'chunks_strict_and_to_example_startfrom1100_biggerthan3000_2048sq.pt')

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
        
        # Initialize list to store feature tensors
        all_features = []
        
        for batch_idx in tqdm(range(num_batches), desc=f"Seq_len {seq_len}"):
            counter += 1
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            
            batch_chunks = chunks[start_idx:end_idx]
            batch_example_idxs = chunk_to_example[start_idx:end_idx]

            with torch.no_grad():
                outputs = model(batch_chunks, output_hidden_states=True)
                logits = outputs.logits
                
                # Get the hidden states from the last layer
                last_hidden_state = outputs.hidden_states[-1]
                
                # Extract the feature tensor for the last token
                # Shape: (batch_size, 1, hidden_size)
                last_token_features = last_hidden_state[:, -2:-1, :].to('cpu')
                
                # Append to our lists
                all_features.append(last_token_features)
                
                # shift_logits = logits[:, -2:-1, :].contiguous()
                # shift_labels = batch_chunks[:, -1:].contiguous()
                
                # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # seq_losses = loss.view(-1)  # Flatten to 1D

                # # Update sums and counts on GPU using scatter_add_
                # per_example_sum_CE.scatter_add_(0, batch_example_idxs, seq_losses)
                # per_example_count.scatter_add_(0, batch_example_idxs, torch.ones_like(seq_losses))

            if args.max_iter > 0:
                if counter > args.max_iter:
                    break

        # Concatenate all features and save
        if all_features:  # Check if we have any features to save
            all_features_tensor = torch.cat(all_features, dim=0)
            all_features_tensor = all_features_tensor.cpu()
            
            # Save the concatenated features and their corresponding example IDs
            save_dict = {
                'features': all_features_tensor.cpu(),
            }
            save_path = os.path.join(SAVE_DIR, f'feature_tensors_seq_{seq_len}.pth')
            torch.save(save_dict, save_path)
            
            # Clear lists to free memory
            del all_features
            torch.cuda.empty_cache()
            print(f"Finish saving for sequence length {seq_len}")

        # Move to CPU only at the end for final calculations
        # mask = per_example_count > 0
        # valid_perplexities = per_example_sum_CE[mask] / per_example_count[mask]
        # valid_counts = per_example_count[mask]
        
        # # Calculate final average
        # if len(valid_perplexities) > 0:
        #     average_perplexity = (valid_perplexities * valid_counts).sum() / valid_counts.sum()
        #     print(f"Average Perplexity for seq_len {seq_len}: {average_perplexity.item()}")
        # else:
        #     print(f"No valid perplexity values computed for seq_len {seq_len}.")

if __name__ == '__main__':
    main()