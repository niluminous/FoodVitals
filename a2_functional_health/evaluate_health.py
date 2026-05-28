import os
import pandas as pd
import numpy as np
import re
import string
import torch
import nltk
from tqdm import tqdm
from bert_score import score as bert_score_func

if not hasattr(np, 'float'):
    np.float = float

os.environ['MOVERSCORE_MODEL'] = "distilbert-base-uncased"

from moverscore_v2 import get_idf_dict, word_mover_score
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. NORMALIZATION & STRICT EVALUATION
# ==========================================
def normalize_yes_no(text):
    if pd.isna(text): return ""
    clean = str(text).lower().translate(str.maketrans('', '', string.punctuation)).strip()
    return clean

def parse_mcq_answer(text):
    if pd.isna(text): return ""
    text = str(text).strip()
    match_explicit = re.search(r'Answer\s*[:\-]\s*([A-D])', text, re.IGNORECASE)
    if match_explicit: return match_explicit.group(1).upper()
    match_start = re.search(r'^([A-D])(?:\)|\.|,|$|\s)', text, re.IGNORECASE)
    if match_start: return match_start.group(1).upper()
    return ""

def evaluate_strict_rows(df):
    results = []
    for idx, row in df.iterrows():
        q_type = row.get('format', 'Short Answer') 
        pred = str(row.get('llm_prediction', '')).strip()
        truth = str(row.get('ground_truth', '')).strip()
        
        is_correct = 0
        
        if q_type == "MCQ":
            p_letter = parse_mcq_answer(pred)
            t_letter = parse_mcq_answer(truth)
            if not t_letter and truth.upper() in ['A','B','C','D']:
                t_letter = truth.upper()
            if p_letter and t_letter and p_letter == t_letter:
                is_correct = 1
                
        elif q_type == "Yes/No":
            p_norm = normalize_yes_no(pred)
            t_norm = normalize_yes_no(truth)
            if p_norm == t_norm:
                is_correct = 1
            elif p_norm.split() and t_norm in ['yes', 'no']:
                if p_norm.split()[0] == t_norm:
                    is_correct = 1
                    
        elif q_type == "Short Answer":
            if normalize_yes_no(pred) == normalize_yes_no(truth):
                is_correct = 1
                
        results.append(is_correct)
    return results

# ==========================================
# 2. SEMANTIC METRICS CALCULATION
# ==========================================
def calculate_semantic_metrics(df_short):
    if df_short.empty: return df_short

    print(f"   > Calculating Semantic Metrics for {len(df_short)} rows...")
    
    cands = [str(x) if pd.notna(x) else "." for x in df_short['llm_prediction']]
    refs = [str(x) if pd.notna(x) else "." for x in df_short['ground_truth']]

    # 1. BERTScore
    P, R, F1 = bert_score_func(
        cands, refs, lang="en", verbose=False, device=DEVICE, 
        batch_size=64, rescale_with_baseline=True
    )
    df_short['bert_score'] = F1.cpu().numpy()
    del P, R, F1
    torch.cuda.empty_cache()

    # 2. MoverScore
    try:
        idf_dict_hyp = get_idf_dict(cands) 
        idf_dict_ref = get_idf_dict(refs)
        scores = word_mover_score(
            refs, cands, idf_dict_ref, idf_dict_hyp, 
            stop_words=stopwords.words('english'), n_gram=1, remove_subwords=True
        )
        df_short['mover_score'] = scores
    except Exception as e:
        print(f"   ! MoverScore Failed: {e}")
        df_short['mover_score'] = 0.0

    # 3. BLEU
    bleu = BLEU(effective_order=True)
    df_short['bleu_score'] = [bleu.sentence_score(c, [r]).score for c, r in zip(cands, refs)]

    # 4. ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    df_short['rouge_l'] = [scorer.score(r, c)['rougeL'].fmeasure for c, r in zip(cands, refs)]

    # 5. METEOR
    df_short['meteor_score'] = [meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(c)) for c, r in zip(cands, refs)]

    return df_short

# ==========================================
# 3. MAIN EXECUTION 
# ==========================================
def run_evaluation(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    filename = os.path.basename(input_csv)
    print(f"\nProcessing: {filename}...")
    
    df = pd.read_csv(input_csv)
    
    # 1. Strict Accuracy
    df['strict_accuracy'] = evaluate_strict_rows(df)

    # 2. Split Data
    short_answer_mask = df['format'] == 'Short Answer'
    df_short = df[short_answer_mask].copy()
    df_others = df[~short_answer_mask].copy()

    # 3. Semantic Metrics
    if not df_short.empty:
        df_short = calculate_semantic_metrics(df_short)
    
    metrics = ['bert_score', 'mover_score', 'bleu_score', 'rouge_l', 'meteor_score']
    for m in metrics:
        df_others[m] = np.nan

    # 4. Merge & Save
    final_df = pd.concat([df_short, df_others]).sort_index()
    final_df.to_csv(output_csv, index=False)
    
    # ==========================================
    # 5. REPORT GENERATION 
    # ==========================================
    print("-" * 80)
    print(f"REPORT: {filename}")
    print("-" * 80)
    
    final_df['__dummy_count__'] = 1
    
    report = final_df.groupby('format').agg({
        'strict_accuracy': 'mean',
        'bert_score': 'mean',
        'mover_score': 'mean',
        'bleu_score': 'mean',
        'rouge_l': 'mean',
        'meteor_score': 'mean',
        '__dummy_count__': 'count'  
    }).rename(columns={'__dummy_count__': 'Count'})

    pd.options.display.float_format = '{:.4f}'.format
    
    print(f"{'Task Type':<15} | {'Count':<5} | {'Acc (Strict)':<12} | {'BERT':<6} | {'Mover':<6} | {'ROUGE':<6} | {'METEOR':<6}")
    print("-" * 80)
    
    for fmt, row in report.iterrows():
        acc = f"{row['strict_accuracy']*100:.2f}%"
        bert = f"{row['bert_score']:.3f}" if not pd.isna(row['bert_score']) else "-"
        mover = f"{row['mover_score']:.3f}" if not pd.isna(row['mover_score']) else "-"
        rouge = f"{row['rouge_l']:.3f}" if not pd.isna(row['rouge_l']) else "-"
        meteor = f"{row['meteor_score']:.3f}" if not pd.isna(row['meteor_score']) else "-"
        print(f"{fmt:<15} | {int(row['Count']):<5} | {acc:<12} | {bert:<6} | {mover:<6} | {rouge:<6} | {meteor:<6}")

    print("-" * 80)
    print(f"Saved to: {output_csv}\n")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    run_evaluation("input_test.csv", "output_test.csv")