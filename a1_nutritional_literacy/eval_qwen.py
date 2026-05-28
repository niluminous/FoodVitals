import pandas as pd
import re
import numpy as np
from scipy.stats import kendalltau

def clean_name(text):
    """Standardizes food names by removing punctuation and extra whitespace."""
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s>]', '', text) 
    return " ".join(text.split())

def get_kendall_tau(truth_str, pred_str):
    try:
        truth_list = [clean_name(x) for x in truth_str.split('>')]
        pred_text_clean = clean_name(pred_str)

        found_positions = []
        for food in truth_list:
            pos = pred_text_clean.find(food)
            if pos != -1:
                found_positions.append((pos, food))
        
        found_positions.sort() 
        pred_items = [item[1] for item in found_positions]

        # Strictness Check: Did the model mention all foods?
        if len(pred_items) != len(truth_list):
            return 0.0

        truth_mapping = {food: i for i, food in enumerate(truth_list)}
        truth_ranks = list(range(len(truth_list)))
        pred_ranks = [truth_mapping[item] for item in pred_items]

        tau, _ = kendalltau(truth_ranks, pred_ranks)
        # Normalize Tau from [-1, 1] to [0, 1]
        return (tau + 1) / 2
    except:
        return 0.0

def evaluate_food_benchmark(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        q_type = row['type']
        prediction = str(row['cleaned_answer']).strip()
        truth = str(row['ground_truth']).strip()
        
        is_correct = 0
        partial_score = np.nan 

        # --- LOGIC: MULTIPLE CHOICE ---
        if q_type == "multiple_choice":
            # Match the letter at the start or surrounded by boundaries
            match = re.search(r'\b([A-D])\b', prediction.upper())
            truth_letter_match = re.search(r'\b([A-D])\b', truth.upper())
            
            if match and truth_letter_match:
                if match.group(1) == truth_letter_match.group(1):
                    is_correct = 1

        # --- LOGIC: RANKING ---
        elif q_type == "ranking":
            def normalize(text):
                return clean_name(text).replace(" ", "")

            # Strict match (ignoring whitespace/case)
            if normalize(truth) == normalize(prediction):
                is_correct = 1
            
            # Continuous metric (Kendall Tau)
            partial_score = get_kendall_tau(truth, prediction)

        # --- LOGIC: QUANTITATIVE RECALL (Yes/No) ---
        elif q_type == "quantitative_recall":
            # Check if prediction starts with ground truth (e.g., "No" matches "No.")
            p_lower = prediction.lower()
            t_lower = truth.lower()
            if p_lower.startswith(t_lower):
                is_correct = 1

        results.append({
            'is_correct': is_correct,
            'kendall_tau_ranking': partial_score
        })

    # Combine results and save
    eval_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    eval_df.to_csv(output_csv, index=False)

    # --- FORMATTED REPORT GENERATION ---
    print("\n" + "="*60)
    print("                NUTRITION BENCHMARK REPORT")
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
    print(f"Full results saved to: {output_csv}\n")

if __name__ == "__main__":
    evaluate_food_benchmark(
        # "qwen3_8b_nutrient_results_cleaned.csv", 
        "a1_nutritional_literacy/qwen3_8b_nutrient_results_cleaned.csv",
        "eval_results_nutrients_qwen8b.csv"
    )