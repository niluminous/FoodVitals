import pandas as pd
import re
import numpy as np
from scipy.stats import kendalltau

def clean_standard(text):
    if pd.isna(text): return ""
    return str(text).lower().strip()

def parse_ranking_to_string(text):

    text = str(text).strip()
    
    # 1. Remove brackets [ ]
    text = re.sub(r'[\[\]]', '', text)
    
    # 2. Remove single or double quotes ' ' or " "
    text = re.sub(r"['\"]", "", text)
    
    # 3. Split and clean
    if ',' in text:
        items = [i.strip().lower() for i in text.split(',')]
    else:
        items = [i.strip().lower() for i in text.split('>')]
    
    # Filter out empty strings in case of trailing commas
    items = [i for i in items if i]
    
    return " > ".join(items)

def get_kendall_tau(truth_str, pred_str):
    try:
        # Convert both to the standardized "a > b > c" format
        t_clean = parse_ranking_to_string(truth_str)
        p_clean = parse_ranking_to_string(pred_str)
        
        truth_list = [x.strip() for x in t_clean.split('>')]
        pred_list = [x.strip() for x in p_clean.split('>')]

        # If lengths don't match, the model missed a food
        if len(truth_list) != len(pred_list):
            return 0.0

        # Mapping truth items to their indices
        truth_mapping = {food: i for i, food in enumerate(truth_list)}
        
        # Convert pred_list to its rank order based on truth
        try:
            pred_ranks = [truth_mapping[food] for food in pred_list]
        except KeyError:
            # Handle case where model hallucinated a food name not in the list
            return 0.0
            
        truth_ranks = list(range(len(truth_list)))
        tau, _ = kendalltau(truth_ranks, pred_ranks)
        
        # Normalized to [0, 1]
        return (tau + 1) / 2
    except:
        return 0.0

def evaluate_food_benchmark(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        q_type = row['type']
        prediction = str(row['llm_prediction']).strip()
        truth = str(row['ground_truth']).strip()
        
        is_correct = 0
        partial_score = np.nan

        # --- LOGIC: MULTIPLE CHOICE ---
        if q_type == "multiple_choice":
            # Extract only the letter (A, B, C, or D)
            p_match = re.search(r'\b([A-D])\b', prediction.upper())
            t_match = re.search(r'\b([A-D])\b', truth.upper())
            
            if p_match and t_match:
                if p_match.group(1) == t_match.group(1):
                    is_correct = 1

        # --- LOGIC: RANKING ---
        elif q_type == "ranking":
            # Convert both to "a > b > c > d" for comparison
            fmt_truth = parse_ranking_to_string(truth)
            fmt_pred = parse_ranking_to_string(prediction)
            
            if fmt_truth == fmt_pred:
                is_correct = 1
            
            partial_score = get_kendall_tau(truth, prediction)

        # --- LOGIC: QUANTITATIVE RECALL ---
        elif q_type == "quantitative_recall":
            p_val = clean_standard(prediction)
            t_val = clean_standard(truth)
            if p_val.startswith(t_val):
                is_correct = 1

        results.append({
            'is_correct': is_correct,
            'kendall_tau_ranking': partial_score
        })

    eval_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    eval_df.to_csv(output_csv, index=False)

    # --- REPORT ---
    print("\n" + "="*60)
    print(f"REPORT FOR: {input_csv}")
    print("="*60)
    
    summary = eval_df.groupby('type').agg({
        'is_correct': 'mean',
        'kendall_tau_ranking': 'mean'
    })
    
    summary.columns = ['Strict Accuracy', 'Avg Kendall Tau']
    
    print(summary.to_string(formatters={
        'Strict Accuracy': '{:,.2%}'.format,
        'Avg Kendall Tau': lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    }))
    print("="*60)

if __name__ == "__main__":
    evaluate_food_benchmark(
        "flan_t5_xl_nutrient_results.csv", 
        "eval_results_flan_t5_xl.csv"
    )