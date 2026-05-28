import pandas as pd
import os
import torch
import time
import subprocess
import re
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


os.environ["VLLM_RPC_TIMEOUT"] = "600000" 
os.environ["VLLM_SHM_BLOCK_TIMEOUT_S"] = "600"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# --- CONFIGURATION ---
INPUT_FILE = "a3_food_safety.jsonl"
OUTPUT_CSV = "qwen8b_food_safety_results.csv"

MODEL_ID = "Qwen/Qwen3-8B"  

def kill_vllm_processes():
    """Cleans up memory for a fresh start."""
    try:
        subprocess.run(['pkill', '-9', '-f', 'vllm'], capture_output=True)
        subprocess.run(['pkill', '-9', '-f', 'multiprocessing'], capture_output=True)
        time.sleep(3) 
    except:
        pass

def get_formatted_prompt(row, tokenizer):
    """
    Constructs a robust prompt specific to Food Safety question formats.
    """
    q_format = row.get('format', 'Short Answer')
    question = row.get('question', '')
    options = row.get('options', '')  
    
    if q_format == "Multiple Choice":
        system_msg = (
            "You are an expert in Food Safety and USDA guidelines. "
            "Analyze the question and the provided options carefully. "
            "Return ONLY the letter corresponding to the correct answer (e.g., 'A', 'B', 'C', or 'D'). "
            "Do not provide the full text of the option."
        )
        # Format the user query to include options clearly
        user_query = f"{question}\n\nOptions:\n{options}"
        
    elif q_format == "Yes/No":
        # Strict instruction for Binary output
        system_msg = (
            "You are an expert in Food Safety. "
            "Answer the following question with ONLY 'Yes' or 'No'. "
            "Do not add punctuation or explanation."
        )
        user_query = question
        
    else: # Short Answer, Fact Retrieval, Scenario-based
        system_msg = (
            "You are an expert in Food Safety. "
            "Provide a direct, high-precision answer. "
            "Your response must be exactly one of the following formats: "
            "1. A short phrase (e.g., 'In a garage'). "
            "2. A comma-separated list (e.g., 'Nausea, vomiting, fever'). "
            "3. A single imperative sentence describing the action (e.g., 'Insert the thermometer into the center'). "
            "Do NOT use introductory text like 'The answer is' or 'You should'."
        )
        user_query = question


    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query}
    ]

    # Apply the chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def parse_qwen_output(output_text):
    """Robustly extracts <think> block and the final response"""
    thinking = ""
    answer = output_text
    
    match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = output_text.split('</think>')[-1].strip()
    elif "<think>" in output_text:
        parts = output_text.split("<think>")
        if len(parts) > 1:
            thinking = parts[1].strip()
            answer = parts[0].strip() 
        else:
            answer = output_text
            
    return thinking, answer.strip()

def main():
    kill_vllm_processes()
    torch.cuda.empty_cache()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    records = df.to_dict('records')
    print(f"Loaded {len(records)} records.")

    print(f"Initializing vLLM with {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading tokenizer (check MODEL_ID): {e}")
        return

    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=1,       
        max_model_len=4096,           
        gpu_memory_utilization=0.80,  
        enforce_eager=True,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=2048,              
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    #  PREPARE PROMPTS
    print(f"Formatting prompts for {len(records)} items...")
    all_prompts = []
    for row in records:
        try:
            p = get_formatted_prompt(row, tokenizer)
            all_prompts.append(p)
        except Exception as e:
            print(f"Skipping row due to formatting error: {e}")
            all_prompts.append("")

    #  RUN INFERENCE
    print("🚀 Starting inference...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    #  PROCESS RESULTS
    batch_results = []
    for j, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        thinking, answer = parse_qwen_output(raw_text)
        row = records[j]
        
        batch_results.append({
            "topic": row.get('topic', ''),
            "category": row.get('category', ''),
            "question_type": row.get('question_type', ''),
            "format": row.get('format', ''),
            "question": row.get('question', ''),
            "options": row.get('options', ''),
            "ground_truth_answer": row.get('ground_truth_answer', ''),
            "evidence_text": row.get('evidence_text', ''),
            "llm_thinking": thinking,
            "llm_prediction": answer
        })

    # SAVE RESULTS
    result_df = pd.DataFrame(batch_results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Success! Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()