import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import torch
import json
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# --- CONFIGURATION ---
INPUT_FILE = "a3_food_safety.jsonl"
OUTPUT_CSV = "flan_t5_xl_food_safety_results.csv"
MODEL_ID = "google/flan-t5-xl"
BATCH_SIZE = 4  
MAX_SEQ_LENGTH = 512

def get_formatted_prompt(row):
    """
    Maintains the 'Expert' persona but re-formats it for Flan-T5.
    Avoids numbered lists to prevent indexing errors.
    """
    q_format = row.get('format', 'Short Answer')
    question = row.get('question', '')
    options = row.get('options', '')

    if q_format == "Multiple Choice":
        prompt = (
            f"You are an expert in Food Safety and USDA guidelines. "
            f"Based on the following options, what is the correct letter answer?\n\n"
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            f"Answer with only the letter:"
        )
    elif q_format == "Yes/No":
        prompt = (
            f"You are an expert in Food Safety. Answer the following question "
            f"with only 'Yes' or 'No'.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    else: # Short Answer / Scenario / Fact Retrieval
        prompt = (
            f"You are an expert in Food Safety. Provide a direct, high-precision "
            f"answer as a short phrase or single sentence. Do not use "
            f"introductory text.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    return prompt

def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    records = df.to_dict('records')
    print(f"Loaded {len(records)} records.")

    print(f"Initializing {MODEL_ID} on {device}...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float32 
    ).to(device)
    model.eval()

    batch_results = []
    print(f"🚀 Starting inference for {len(records)} items...")

    for i in range(0, len(records), BATCH_SIZE):
        batch_chunk = records[i : i + BATCH_SIZE]
        prompts = [get_formatted_prompt(row) for row in batch_chunk]

        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5, 
                    early_stopping=True
                )

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, prediction in enumerate(decoded_outputs):
                row = batch_chunk[j]
                
                batch_results.append({
                    "topic": row.get('topic', ''),
                    "category": row.get('category', ''),
                    "question_type": row.get('question_type', ''),
                    "format": row.get('format', ''),
                    "question": row.get('question', ''),
                    "options": row.get('options', ''),
                    "ground_truth_answer": row.get('ground_truth_answer', ''),
                    "evidence_text": row.get('evidence_text', ''),
                    "llm_prediction": prediction.strip()
                })
            

            if (i // BATCH_SIZE) % 10 == 0:
                print(f"Processed {i + len(batch_chunk)}/{len(records)}...")

        except RuntimeError as e:
            print(f"Error encountered at index {i}: {e}")
            torch.cuda.empty_cache()
            continue

    # SAVE RESULTS
    result_df = pd.DataFrame(batch_results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Success! Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()



# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# import pandas as pd
# import torch
# import json
# from transformers import T5TokenizerFast, T5ForConditionalGeneration

# # --- CONFIGURATION ---
# INPUT_FILE = "food_safety.jsonl"
# OUTPUT_CSV = "flan_t5_large_food_safety_results.csv" 
# MODEL_ID = "google/flan-t5-large" 
# BATCH_SIZE = 8  
# MAX_SEQ_LENGTH = 512

# def get_formatted_prompt(row):
#     """
#     Maintains the 'Expert' persona but re-formats it for Flan-T5.
#     Avoids numbered lists to prevent indexing errors.
#     """
#     q_format = row.get('format', 'Short Answer')
#     question = row.get('question', '')
#     options = row.get('options', '')

#     if q_format == "Multiple Choice":
#         prompt = (
#             f"You are an expert in Food Safety and USDA guidelines. "
#             f"Based on the following options, what is the correct letter answer?\n\n"
#             f"Question: {question}\n"
#             f"Options: {options}\n\n"
#             f"Answer with only the letter:"
#         )
#     elif q_format == "Yes/No":
#         prompt = (
#             f"You are an expert in Food Safety. Answer the following question "
#             f"with only 'Yes' or 'No'.\n\n"
#             f"Question: {question}\n"
#             f"Answer:"
#         )
#     else: # Short Answer / Scenario / Fact Retrieval
#         prompt = (
#             f"You are an expert in Food Safety. Provide a direct, high-precision "
#             f"answer as a short phrase or single sentence. Do not use "
#             f"introductory text.\n\n"
#             f"Question: {question}\n"
#             f"Answer:"
#         )
#     return prompt

# def main():
#     torch.cuda.empty_cache()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     if not os.path.exists(INPUT_FILE):
#         print(f"Error: {INPUT_FILE} not found.")
#         return

#     print(f"Loading data from {INPUT_FILE}...")
#     df = pd.read_json(INPUT_FILE, lines=True)
#     records = df.to_dict('records')

#     print(f"Initializing {MODEL_ID} on {device}...")
#     tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    

#     model = T5ForConditionalGeneration.from_pretrained(
#         MODEL_ID, 
#         torch_dtype=torch.float32 
#     ).to(device)
#     model.eval()

#     batch_results = []
#     print(f"🚀 Starting inference for {len(records)} items...")

#     for i in range(0, len(records), BATCH_SIZE):
#         batch_chunk = records[i : i + BATCH_SIZE]
#         prompts = [get_formatted_prompt(row) for row in batch_chunk]

#         inputs = tokenizer(
#             prompts, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=MAX_SEQ_LENGTH
#         ).to(device)

#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=50,
#                     num_beams=5, 
#                     early_stopping=True
#                 )

#             decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#             for j, prediction in enumerate(decoded_outputs):
#                 row = batch_chunk[j]
                

#                 batch_results.append({
#                     "topic": row.get('topic', ''),
#                     "category": row.get('category', ''),
#                     "question_type": row.get('question_type', ''),
#                     "format": row.get('format', ''),
#                     "question": row.get('question', ''),
#                     "options": row.get('options', ''),
#                     "ground_truth_answer": row.get('ground_truth_answer', ''),
#                     "evidence_text": row.get('evidence_text', ''),
#                     "llm_prediction": prediction.strip()
#                 })
            
#             if (i // BATCH_SIZE) % 10 == 0:
#                 print(f"Processed {i + len(batch_chunk)}/{len(records)}...")

#         except RuntimeError as e:

#             print(f"Error encountered at index {i}: {e}")
#             torch.cuda.empty_cache()
#             continue


#     result_df = pd.DataFrame(batch_results)
#     result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
#     print(f"✅ Success! Results saved to {OUTPUT_CSV}")

# if __name__ == '__main__':
#     main()