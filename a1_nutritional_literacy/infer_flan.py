import pandas as pd
import os
import torch
import time
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# --- CONFIGURATION ---
INPUT_CSV = "nutrients_task_qa_shuffled.csv"
OUTPUT_CSV = "flan_t5_xl_nutrient_results.csv"
MODEL_ID = "google/flan-t5-xl"
# MODEL_ID = "google/flan-t5-xl"
BATCH_SIZE = 4  
MAX_SEQ_LENGTH = 512  

def get_formatted_prompt(row):
    q_type = row['type']
    user_query = row['prompt']
    if q_type == "multiple_choice":
        instruction = "Answer with ONLY the correct option letter and the food name (e.g., 'A) Apple')."
    elif q_type == "ranking":
        instruction = "Rank the following foods: [A, B, C, D] Constraint: Your answer must be a single line containing all four foods separated by '>'."
    elif q_type == "quantitative_recall":
        instruction = "Answer with ONLY 'Yes' or 'No'."
    else:
        instruction = "Answer accurately based on nutritional science."

    return f"{instruction}\n\nQ: {user_query}\nA: "

def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """Safely decode token IDs with bounds checking"""
    vocab_size = tokenizer.vocab_size
    if token_ids is None:
        return ""
    

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    
    if not isinstance(token_ids, list) or len(token_ids) == 0:
        return ""

    if isinstance(token_ids[0], list):
        decoded_texts = []
        for seq in token_ids:
            valid_tokens = []
            for token_id in seq:
                if isinstance(token_id, (int, float)) and 0 <= token_id < vocab_size:
                    valid_tokens.append(int(token_id))
                else:
                    valid_tokens.append(tokenizer.unk_token_id)
            try:
                decoded = tokenizer.decode(valid_tokens, skip_special_tokens=skip_special_tokens)
                decoded_texts.append(decoded)
            except Exception as e:
                print(f"Decoding error in batch: {e}")
                decoded_texts.append("[ERROR: DECODING_FAILED]")
        return decoded_texts
    else:
        valid_tokens = []
        for token_id in token_ids:
            if isinstance(token_id, (int, float)) and 0 <= token_id < vocab_size:
                valid_tokens.append(int(token_id))
            else:
                valid_tokens.append(tokenizer.unk_token_id)
        try:
            return tokenizer.decode(valid_tokens, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            print(f"Decoding error: {e}")
            return "[ERROR: DECODING_FAILED]"

def main():

    torch.cuda.empty_cache()
    
    print(f"Loading benchmark from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df = df.sort_values(by='type') 

    if os.path.exists(OUTPUT_CSV):
        processed_df = pd.read_csv(OUTPUT_CSV)
        processed_prompts = set(processed_df['prompt'])
        df_to_process = df[~df['prompt'].isin(processed_prompts)].copy()
        print(f"Resuming: Skipping {len(df) - len(df_to_process)} already processed items.")
    else:
        df_to_process = df.copy()

    if len(df_to_process) == 0:
        print("All tasks completed.")
        return


    print(f"Initializing {MODEL_ID}...")
    
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float32  
    )
    
    
    model = model.to(device)
    model.eval()  
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    records = df_to_process.to_dict('records')
    final_results = []
    current_batch_size = BATCH_SIZE

    print(f"Processing {len(records)} questions with batch size {current_batch_size}...")

    i = 0
    while i < len(records):
            batch_chunk = records[i:i + current_batch_size]
            prompts = [get_formatted_prompt(row) for row in batch_chunk]
            
            first_item_type = batch_chunk[0]['type']
            
            if first_item_type == "ranking":
                min_tokens, max_tokens, penalty = 16, 128, 1.0
                rep_penalty = 1.0  
            elif first_item_type == "multiple_choice":
                min_tokens, max_tokens, penalty = 1, 20, 1.0
                rep_penalty = 1.0
            else: # quantitative_recall
                min_tokens, max_tokens, penalty = 1, 5, 1.0
                rep_penalty = 1.0

            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
                

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        min_new_tokens=min_tokens,
                        length_penalty=penalty,
                        repetition_penalty=rep_penalty,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        return_dict_in_generate=True  
                    )

                generated_sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs

                generated_sequences_cpu = generated_sequences.cpu()

                decoded_outputs = []
                for seq in generated_sequences_cpu:
                    decoded_text = safe_decode(tokenizer, seq)
                    decoded_outputs.append(decoded_text)
                batch_results = []
                for j, generated_text in enumerate(decoded_outputs):
                    row = batch_chunk[j]
                    batch_results.append({
                        "type": row['type'],
                        "nutrient": row['nutrient'],
                        "prompt": row['prompt'],
                        "ground_truth": row['ground_truth'],
                        "llm_prediction": str(generated_text).strip(),
                    })
                
                pd.DataFrame(batch_results).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                i += current_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                i += current_batch_size

if __name__ == '__main__':
    main()
