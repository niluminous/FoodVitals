import pandas as pd
import os
import torch
import time
import subprocess
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["VLLM_RPC_TIMEOUT"] = "1200000"        
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "600"   
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

INPUT_CSV = "nutrients_task_qa.csv"
OUTPUT_CSV = "qwen32B_nutrient_results.csv"
MODEL_ID = "Qwen/Qwen3-32B-AWQ"
BATCH_SIZE = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def kill_vllm_processes():
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'multiprocessing'], capture_output=True)
        time.sleep(5) 
    except:
        pass

def get_formatted_prompt(row):
    q_type = row['type']
    user_query = row['prompt']
    

    if q_type == "multiple_choice":
        system_msg = "You are a nutritional data expert. Please show your choice in the 'answer' field with only the choice letter, e.g., 'answer': 'C'."
    elif q_type == "ranking":
        system_msg = "Rank the foods from highest to lowest nutrient content. Use the '>' symbol.  Provide ONLY the final ranked names (e.g., 'Apple > Pear > Grape')."
    else:
        system_msg = "Answer the following nutritional question accurately. Answer 'Yes' or 'No' where appropriate."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # This triggers the <think> mode
    )

def parse_qwen_output(output_text):
    """Separates the <think> block from the final answer"""
    try:
        if "</think>" in output_text:
            parts = output_text.split("</think>")
            thinking = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
            return thinking, answer
        return "", output_text.strip()
    except:
        return "", output_text

def main():
    kill_vllm_processes()
    torch.cuda.empty_cache()
    
    print(f"Loading benchmark from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    llm = LLM(
            model=MODEL_ID, 
            tensor_parallel_size=2, 
            max_model_len=8192,           
            gpu_memory_utilization=0.85, 
            enforce_eager=True,
            quantization="awq",
            disable_custom_all_reduce=True 
        )

    sampling_params = SamplingParams(
        temperature=0.0, 
        # top_p=0.95, 
        # top_k=20,
        # presence_penalty=1.5,         
        max_tokens=4096               
    )

    records = df.to_dict('records')
    chunks = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]

    for i, chunk in enumerate(chunks):
        batch_prompts = [get_formatted_prompt(row) for row in chunk]
        outputs = llm.generate(batch_prompts, sampling_params)
        
        batch_results = []
        for j, output in enumerate(outputs):
            raw_text = output.outputs[0].text
            thinking, answer = parse_qwen_output(raw_text)
            row = chunk[j]
            
            batch_results.append({
                "type": row['type'],
                "nutrient": row['nutrient'],
                "prompt": row['prompt'],
                "ground_truth": row['ground_truth'],
                "llm_thinking": thinking,      
                "llm_prediction": answer,    
                "verification_data": row['verification_data']
            })
        
        chunk_df = pd.DataFrame(batch_results)
        header = not os.path.exists(OUTPUT_CSV)
        chunk_df.to_csv(OUTPUT_CSV, mode='a', header=header, index=False)
        print(f"Batch {i+1}/{len(chunks)} saved.")

if __name__ == '__main__':
    main()





