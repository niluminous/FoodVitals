

import pandas as pd
import os
import torch
import time
import subprocess
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# PERFORMANCE & STABILITY ENVIRONMENT VARIABLES 
os.environ["VLLM_RPC_TIMEOUT"] = "600000" 
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "300" 
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

#  CONFIGURATION 
INPUT_CSV = "balanced_health_QA_shuffled.csv"
OUTPUT_CSV = "llama31_8b_health_claims_results.csv"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
BATCH_SIZE = 16 

def kill_vllm_processes():
    """Clean up any lingering vLLM processes"""
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'multiprocessing'], capture_output=True)
        time.sleep(2)  
    except:
        pass

def get_formatted_prompt(row):
    """Llama-3.1 specific chat template with Health Claim prompt logic."""
    q_type = row['Format'] 
    user_query = row['Generated_Question']
    
    if q_type == "MCQ":
        system_msg = (
            "You are an expert in food health claims and regulatory science. "
            "Please provide only the correct choice letter (e.g., 'A', 'B', 'C', or 'D')."
        )
    elif q_type == "Yes/No":
        system_msg = "You are a health claims expert. Answer the following question with only 'Yes' or 'No'."
    elif q_type == "Short Answer":
        system_msg = (
            "You are a health claims expert. Provide a concise, scientifically accurate response. "
            "This can be a specific phrase (e.g., 'Cell division and specialization') or a "
            "formal sentence (e.g., 'Contributes to the maintenance of normal brain function'). "
            "Use exact regulatory terminology like 'supports', 'needed for', or 'contributes to'."
        )
    else:
        system_msg = "Answer the following health-related question accurately."

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def main():
    kill_vllm_processes()
    torch.cuda.empty_cache()
    
    print(f"Loading benchmark from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if os.path.exists(OUTPUT_CSV):
        processed_df = pd.read_csv(OUTPUT_CSV)
        processed_questions = set(processed_df['question'])
        df_to_process = df[~df['Generated_Question'].isin(processed_questions)].copy()
        print(f"Resuming: Skipping {len(df) - len(df_to_process)} already processed items.")
    else:
        df_to_process = df.copy()

    if len(df_to_process) == 0:
        print("All tasks completed.")
        return

    # INITIALIZE vLLM 
    print(f"Initializing vLLM with {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=1,       
        max_model_len=4096,           
        gpu_memory_utilization=0.90,   
        enforce_eager=True,           
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=256,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

    # --- RUN INFERENCE ---
    records = df_to_process.to_dict('records')
    all_prompts = [get_formatted_prompt(row) for row in records]
    
    print(f"Processing {len(all_prompts)} questions...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    final_results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        row = records[i]
        
        final_results.append({
            "food_substance": row['Food'],
            "format": row['Format'],
            "original_claim": row['Original_Claim'],
            "target_population": row['Target_Population'],
            "question": row['Generated_Question'],
            "ground_truth": row['Ground_Truth'],
            "llm_prediction": generated_text
        })
        
        if (i + 1) % 50 == 0:
            temp_df = pd.DataFrame(final_results)
            header = not os.path.exists(OUTPUT_CSV)
            temp_df.to_csv(OUTPUT_CSV, mode='a', header=header, index=False, encoding='utf-8-sig')
            final_results = [] 
            print(f"Processed {i+1}/{len(all_prompts)} rows...")

    if final_results:
        temp_df = pd.DataFrame(final_results)
        header = not os.path.exists(OUTPUT_CSV)
        temp_df.to_csv(OUTPUT_CSV, mode='a', header=header, index=False, encoding='utf-8-sig')

    print(f"Inference complete! Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()



