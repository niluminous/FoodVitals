import os
import re
import pandas as pd
import numpy as np


if not hasattr(np, 'float'):
    np.float = float 

import nltk
from evaluate import load
from bert_score import score as bert_score_func

os.environ['MOVERSCORE_MODEL'] = "distilbert-base-uncased"
from moverscore_v2 import get_idf_dict, word_mover_score

# # Download NLTK resources
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

def robust_mcq_extraction(text):
    """Rigorous extraction of MCQ letters (A-D)."""
    if not isinstance(text, str): return ""
    text = text.strip().upper()
    # Check start of string (e.g., "A) ...")
    start_match = re.match(r'^([A-D])(?:\W|$)', text)
    if start_match: return start_match.group(1)
    # Check for "Answer: A" or similar
    pattern_match = re.search(r'(?:ANSWER|OPTION|LETTER)\s*:?\s*([A-D])', text)
    if pattern_match: return pattern_match.group(1)
    # First isolated letter
    fallback_match = re.search(r'\b([A-D])\b', text)
    if fallback_match: return fallback_match.group(1)
    return text

def normalize_yes_no(text):
    """Robust Yes/No normalization."""
    if not isinstance(text, str): return ""
    clean = text.lower().strip()
    if 'yes' in clean: return 'yes'
    if 'no' in clean: return 'no'
    return clean

def evaluate_food_safety(input_csv):
    df = pd.read_csv(input_csv)
    results = {}

    # --- MCQ and Yes/No Accuracy ---
    for fmt in ["Multiple Choice", "Yes/No"]:
        subset = df[df['format'] == fmt].copy()
        if not subset.empty:
            if fmt == "Multiple Choice":
                subset['gt_clean'] = subset['ground_truth_answer'].apply(robust_mcq_extraction)
                subset['pred_clean'] = subset['llm_prediction'].apply(robust_mcq_extraction)
            else:
                subset['gt_clean'] = subset['ground_truth_answer'].apply(normalize_yes_no)
                subset['pred_clean'] = subset['llm_prediction'].apply(normalize_yes_no)
            
            acc = (subset['gt_clean'] == subset['pred_clean']).mean()
            results[f"{fmt} Accuracy"] = acc

    # Short Answer Metrics 
    sa_subset = df[df['format'].isin(['Short Answer', 'Scenario-based'])]
    if not sa_subset.empty:
        refs = sa_subset['ground_truth_answer'].tolist()
        preds = sa_subset['llm_prediction'].tolist()

        # BLEU, ROUGE, METEOR
        for m_name in ['bleu', 'rouge', 'meteor']:
            metric = load(m_name)
            if m_name == 'rouge':
                results['ROUGE-L'] = metric.compute(predictions=preds, references=refs)['rougeL']
            elif m_name == 'meteor':
                results['METEOR'] = metric.compute(predictions=preds, references=refs)['meteor']
            else:
                results['BLEU'] = metric.compute(predictions=preds, references=refs)['bleu']

        # BERTScore (Rescale = True)
        P, R, F1 = bert_score_func(preds, refs, lang="en", rescale_with_baseline=True)
        results['BERTScore-F1'] = F1.mean().item()

        # MoverScore 
        idf_dict_hyp = get_idf_dict(preds)
        idf_dict_ref = get_idf_dict(refs)
        m_scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, 
                                    stop_words=[], n_gram=1, remove_subwords=True)
        results['MoverScore'] = np.mean(m_scores)

    # --- PRINTING ---
    print("\n" + "="*45 + "\nFOOD SAFETY EVALUATION RESULTS\n" + "="*45)
    for k, v in results.items():
        print(f"{k:25} | {v:.2%}" if "Accuracy" in k else f"{k:25} | {v:.4f}")

if __name__ == "__main__":
    FILE_TO_EVAL = "flan_t5_xl_food_safety_results.csv"
    evaluate_food_safety(FILE_TO_EVAL)