import pandas as pd
import os
import torch
import time
import subprocess
from vllm import LLM, SamplingParams

os.environ["VLLM_RPC_TIMEOUT"] = "600000"        
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "300"    
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

INPUT_CSV = "nutrients_task_qa_shuffled.csv"
OUTPUT_CSV = "llama33_benchmark_results.csv"
MODEL_ID = "casperhansen/llama-3.3-70b-instruct-awq"
BATCH_SIZE = 4

def kill_vllm_processes():
    """Clean up any lingering vLLM processes"""
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'multiprocessing'], capture_output=True)
        time.sleep(2)  
    except:
        pass

def get_formatted_prompt(row):
    q_type = row['type']
    user_query = row['prompt']
    
    if q_type == "multiple_choice":
        system_msg = "You are a nutritional data expert. Answer with ONLY the correct option letter and the food name (e.g., 'A) Apple')."
    elif q_type == "ranking":
        system_msg = "Rank the foods from highest to lowest nutrient content. Use the '>' symbol. Provide ONLY the final ranked names (e.g., 'Apple > Pear > Grape')."
    elif q_type == "quantitative_recall":
        system_msg = "Answer the following threshold question with ONLY 'Yes' or 'No'."
    else:
        system_msg = "Answer the question accurately based on nutritional science."

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
        processed_prompts = set(processed_df['prompt'])
        df_to_process = df[~df['prompt'].isin(processed_prompts)].copy()
        print(f"Resuming: Skipping {len(df) - len(df_to_process)} already processed items.")
    else:
        df_to_process = df.copy()

    if len(df_to_process) == 0:
        print("All tasks completed.")
        return
    
    print(f"Initializing vLLM with {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=2, 
        max_model_len=4096,           # Cap context to save KV cache memory
        gpu_memory_utilization=0.85,  # Leave room for system overhead
        enforce_eager=True,           # Skip slow compilation hangs
        disable_custom_all_reduce=True # Force standard, stable communication
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)

    # --- RUN INFERENCE ---
    records = df_to_process.to_dict('records')
    chunks = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]

    print(f"Processing {len(records)} questions in {len(chunks)} batches...")

    for i, chunk in enumerate(chunks):
        batch_prompts = [get_formatted_prompt(row) for row in chunk]
        outputs = llm.generate(batch_prompts, sampling_params)
        
        batch_results = []
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            row = chunk[j]
            
            batch_results.append({
                "type": row['type'],
                "nutrient": row['nutrient'],
                "prompt": row['prompt'],
                "ground_truth": row['ground_truth'],
                "llm_prediction": generated_text,
                "verification_data": row['verification_data']
            })
        
        chunk_df = pd.DataFrame(batch_results)
        header = not os.path.exists(OUTPUT_CSV)
        chunk_df.to_csv(OUTPUT_CSV, mode='a', header=header, index=False)
        print(f"Batch {i+1}/{len(chunks)} saved.")

if __name__ == '__main__':
    main()
