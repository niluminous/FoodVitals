import pandas as pd
import re
import numpy as np
from scipy.stats import kendalltau
import string

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

        if len(pred_items) != len(truth_list):
            return 0.0

        truth_mapping = {food: i for i, food in enumerate(truth_list)}
        truth_ranks = list(range(len(truth_list)))
        pred_ranks = [truth_mapping[item] for item in pred_items]

        tau, _ = kendalltau(truth_ranks, pred_ranks)
        return (tau + 1) / 2
    except:
        return 0.0

def evaluate_food_benchmark(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        q_type = row['type']
        prediction = str(row['cleaned_prediction']).strip()
        # prediction = str(row['cleaned_prediction']).strip()
        truth = str(row['ground_truth']).strip()
        
        is_correct = 0
        partial_score = np.nan  


        if q_type == "multiple_choice":

            match = re.search(r'^\s*([A-D])(?:\b|[\)\.\_\-\s])', prediction.upper())
            truth_letter_match = re.search(r'\b([A-D])\b', truth.upper())
            
            if match and truth_letter_match:
                if match.group(1) == truth_letter_match.group(1):
                    is_correct = 1
        # --- LOGIC: RANKING ---
        elif q_type == "ranking":
            clean_truth = clean_name(truth).replace(" ", "")
            clean_pred = clean_name(prediction).replace(" ", "")
            if clean_truth == clean_pred:
                is_correct = 1
            # Calculate partial score ONLY for ranking
            partial_score = get_kendall_tau(truth, prediction)

        # --- LOGIC: QUANTITATIVE RECALL (Yes/No) ---
        elif q_type == "quantitative_recall":
            # p_lower = prediction.lower()
            # t_lower = truth.lower()
            # if p_lower.startswith(t_lower):
            #     is_correct = 1
            # Remove all punctuation (.,?!) from prediction
            p_clean = prediction.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            t_clean = truth.lower().strip()
            p_words = p_clean.split()
            # Check for exact word match (split() handles "No logic needed")
            # This checks if the first word is exactly "no" or "yes"
            if p_words and p_words[0] == t_clean:
                is_correct = 1

        results.append({
            'is_correct': is_correct,
            'kendall_tau_ranking': partial_score
        })

    # Combine results
    eval_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    eval_df.to_csv(output_csv, index=False)

    # CLEAN REPORT 
    print("\n" + "="*50)
    print("           NUTRITION BENCHMARK REPORT")
    print("="*50)
    
    # Calculate Mean Accuracy for all
    summary = eval_df.groupby('type').agg({
        'is_correct': 'mean',
        'kendall_tau_ranking': 'mean' 
    })
    
    summary.columns = ['Strict Accuracy', 'Avg Kendall Tau (Ranking Only)']
    
    # Formatting the output table
    print(summary.to_string(formatters={
        'Strict Accuracy': '{:,.2%}'.format,
        'Avg Kendall Tau (Ranking Only)': lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
    }))
    print("="*50)

if __name__ == "__main__":
    evaluate_food_benchmark("llama31_8b_nutrient_results.csv", "eval_results_nutrients_llama8B.csv")