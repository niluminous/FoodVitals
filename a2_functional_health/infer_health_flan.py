

import pandas as pd
import os
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

INPUT_CSV = "balanced_health_QA_shuffled.csv"
OUTPUT_CSV = "flan_t5_large_health_results.csv"
MODEL_ID = "google/flan-t5-large"  
BATCH_SIZE = 16  
MAX_SEQ_LENGTH = 512 

def get_formatted_prompt(row):
    q_type = str(row['Format']).strip()
    user_query = row['Generated_Question']
    
    if q_type == "MCQ":
        instruction = "Answer with ONLY the correct option letter (e.g., 'A')."
    elif q_type == "Yes/No":
        instruction = "Answer with ONLY 'Yes' or 'No'."
    elif q_type == "Short Answer":
        instruction = "Provide a concise, scientifically accurate health claim response."
    else:
        instruction = "Answer accurately."

    return f"Instruction: {instruction}\nQuestion: {user_query}\nAnswer:"

def main():
    torch.cuda.empty_cache()
    
    df = pd.read_csv(INPUT_CSV)
    
    if os.path.exists(OUTPUT_CSV):
        processed_df = pd.read_csv(OUTPUT_CSV)
        processed_qs = set(processed_df['question'].astype(str))
        df_to_process = df[~df['Generated_Question'].astype(str).isin(processed_qs)].copy()
    else:
        df_to_process = df.copy()

    if len(df_to_process) == 0:
        return

    print(f"Initializing {MODEL_ID}...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    records = df_to_process.to_dict('records')
    
    print(f"Processing {len(records)} questions...")

    for i in range(0, len(records), BATCH_SIZE):
        batch_chunk = records[i:i + BATCH_SIZE]
        prompts = [get_formatted_prompt(row) for row in batch_chunk]
        
        q_format = batch_chunk[0]['Format']
        max_toks = 64 if q_format == "Short Answer" else 10

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_toks,
                num_beams=5,
                early_stopping=True
            )

        batch_results = []
        for j, seq in enumerate(outputs):
            pred_text = tokenizer.decode(seq, skip_special_tokens=True)
            row = batch_chunk[j]
            
            batch_results.append({
                "food": row['Food'],
                "format": row['Format'],
                "question": row['Generated_Question'],
                "ground_truth": row['Ground_Truth'],
                "llm_prediction": pred_text.strip()
            })
        
        pd.DataFrame(batch_results).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
        
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"Batch {i//BATCH_SIZE} completed...")

    print(f"✅ Finished! Results in {OUTPUT_CSV}")

if __name__ == '__main__':
    main()