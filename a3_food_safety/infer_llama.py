import pandas as pd
import os
import torch
import time
import subprocess
import json
from vllm import LLM, SamplingParams

# --- STABILITY SETTINGS FOR LARGE MODELS ---
os.environ["VLLM_RPC_TIMEOUT"] = "600000" 
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "600"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# --- CONFIGURATION ---
INPUT_FILE = "a3_food_safety.jsonl"
OUTPUT_CSV = "llama33_70b_awq_food_safety_results.csv"
MODEL_ID = "casperhansen/llama-3.3-70b-instruct-awq"

def kill_vllm_processes():
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'multiprocessing'], capture_output=True)
        time.sleep(5) 
    except:
        pass

def get_llama_prompt(row):
    """Exactly the same prompt logic as your 8B/Qwen setup"""
    q_format = row.get('format', 'Short Answer')
    question = row.get('question', '')
    options = row.get('options', '')
    
    if q_format == "Multiple Choice":
        system_msg = "You are an expert in Food Safety and USDA guidelines. Analyze the question and the provided options carefully. Return ONLY the letter corresponding to the correct answer (e.g., 'A', 'B', 'C', or 'D'). Do not provide the full text of the option."
        user_query = f"{question}\n\nOptions:\n{options}"
    elif q_format == "Yes/No":
        system_msg = "You are an expert in Food Safety. Answer the following question with ONLY 'Yes' or 'No'. Do not add punctuation or explanation."
        user_query = question
    else:
        system_msg = "You are an expert in Food Safety. Provide a direct, high-precision answer. Your response must be exactly one of the following formats: 1. A short phrase. 2. A comma-separated list. 3. A single imperative sentence describing the action. Do NOT use introductory text."
        user_query = question

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def main():
    kill_vllm_processes()
    torch.cuda.empty_cache()
    
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    
    if os.path.exists(OUTPUT_CSV):
        processed_df = pd.read_csv(OUTPUT_CSV)
        processed_questions = set(processed_df['question'])
        df = df[~df['question'].isin(processed_questions)].copy()
        print(f"Resuming: Skipping {len(processed_questions)} items.")

    if len(df) == 0:
        print("All tasks completed.")
        return

    records = df.to_dict('records')

    # 2. INITIALIZE vLLM 
    print(f"Initializing vLLM with {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=2,       
        max_model_len=4096,            
        gpu_memory_utilization=0.85,   
        quantization="awq",           
        enforce_eager=True,          
        disable_custom_all_reduce=True 
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128, stop=["<|eot_id|>"])

    # RUN INFERENCE 
    all_prompts = [get_llama_prompt(row) for row in records]
    print(f"🚀 Processing {len(all_prompts)} questions...")
    
    outputs = llm.generate(all_prompts, sampling_params)
    
    final_results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        row = records[i]
        
        final_results.append({
            "topic": row.get('topic', ''),
            "format": row.get('format', ''),
            "question": row.get('question', ''),
            "ground_truth_answer": row.get('ground_truth_answer', ''),
            "llm_prediction": generated_text
        })
        
        if (i + 1) % 50 == 0:
            temp_df = pd.DataFrame(final_results)
            header = not os.path.exists(OUTPUT_CSV)
            temp_df.to_csv(OUTPUT_CSV, mode='a', header=header, index=False)
            final_results = [] 
            print(f"Saved batch through item {i+1}...")

    if final_results:
        pd.DataFrame(final_results).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)

    print(f"✅ Inference complete. Final results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
