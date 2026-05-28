import pandas as pd
import os
import torch
import time
import subprocess
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INPUT_CSV = "balanced_health_QA_shuffled.csv" 
OUTPUT_CSV = "qwen3_8b_health_claims_results.csv"
MODEL_ID = "Qwen/Qwen3-8B" 

os.environ["VLLM_RPC_TIMEOUT"] = "600000" 
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "600"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def kill_vllm_processes():
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        time.sleep(3) 
    except:
        pass

def get_formatted_prompt(row):
    q_type = row['Format'] 
    user_query = row['Generated_Question']
    
    if q_type == "MCQ":
        system_msg = (
            "You are a health claims expert. Please identify the correct choice letter "
            "based on approved health claims. Provide only the letter, e.g., 'A'."
        )
    elif q_type == "Yes/No":
        system_msg = "You are a health claims expert. Answer with only 'Yes' or 'No'."
    elif q_type == "Short Answer":
        system_msg = (
            "You are a health claims expert. Provide a concise, scientifically accurate response. "
            "This can be a specific phrase (e.g., 'Cell division and specialization') or a "
            "formal sentence (e.g., 'Contributes to the maintenance of normal brain function'). "
            "Use exact regulatory terminology like 'supports', 'needed for', or 'contributes to'."
        )
    else:
        system_msg = "Answer the following health claim question accurately."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  
    )

def parse_qwen_output(output_text):
    """Extracts <think> block and the final response"""
    thinking = ""
    answer = output_text
    
    match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = output_text.split('</think>')[-1].strip()
    elif "<think>" in output_text:
        parts = output_text.split("<think>")
        answer = parts[0].strip()
        thinking = parts[1].strip()
        
    return thinking, answer

def main():
    kill_vllm_processes()
    torch.cuda.empty_cache()
    
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Initializing vLLM with {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=2048, 
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    records = df.to_dict('records')
    all_prompts = [get_formatted_prompt(row) for row in records]
    
    print(f"Starting inference for {len(all_prompts)} health claim questions...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    batch_results = []
    for j, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        thinking, answer = parse_qwen_output(raw_text)
        row = records[j]
        
        batch_results.append({
            "food_substance": row['Food'],
            "format": row['Format'],
            "original_claim": row['Original_Claim'],
            "target_population": row['Target_Population'],
            "question": row['Generated_Question'],
            "ground_truth": row['Ground_Truth'],
            "llm_thinking": thinking,
            "llm_prediction": answer
        })

    result_df = pd.DataFrame(batch_results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

