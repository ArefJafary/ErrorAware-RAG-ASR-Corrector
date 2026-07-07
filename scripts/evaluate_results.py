import pandas as pd
import sys
import os
import yaml


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.normalization import EvaluationNormalizer
from src.metrics import calculate_row_wer

def evaluate_pipeline():
    print("\n" + "=" * 60)
    print("Starting Error-Aware RAG Evaluation")
    print("=" * 60)

    # --- Load Configuration ---
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # The input for evaluation is the OUTPUT from your main inference script
    input_file = config['paths']['output_csv']
    output_file = config['paths'].get('metrics_csv', 'data/processed/final_test_metrics.csv')

    print(f"Loading test predictions from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Cannot find '{input_file}'. Please run 'main.py' first.")
        sys.exit(1)

    initial_len = len(df)

    # Using your exact column names!
    cols_to_check = ['reference', 'whisper_base', 'std_rag_output', 'custom_rag_output']
    df = df.dropna(subset=cols_to_check).copy()

    if initial_len - len(df) > 0:
        print(f"Dropped {initial_len - len(df)} null rows. (Remaining: {len(df)} rows)")

    # Apply strict morphological cleaning and ITN directly to the correct columns
    print("Applying Inverse Text Normalization (ITN) and strict morphological stripping...")
    eval_cleaner = EvaluationNormalizer()

    # Convert to string just in case Pandas read any pure numbers as integers
    df['ref_clean'] = df['reference'].astype(str).apply(eval_cleaner.clean)
    df['base_clean'] = df['whisper_base'].astype(str).apply(eval_cleaner.clean)
    df['std_clean'] = df['std_rag_output'].astype(str).apply(eval_cleaner.clean)
    df['cust_clean'] = df['custom_rag_output'].astype(str).apply(eval_cleaner.clean)

    print("Calculating pure WER for each configuration...")
    df['wer_base_new'] = df.apply(lambda row: calculate_row_wer(row['ref_clean'], row['base_clean']), axis=1)
    df['wer_std_new'] = df.apply(lambda row: calculate_row_wer(row['ref_clean'], row['std_clean']), axis=1)
    df['wer_cust_new'] = df.apply(lambda row: calculate_row_wer(row['ref_clean'], row['cust_clean']), axis=1)

    df_eval = df[df['ref_clean'].str.len() > 0]
    avg_base = df_eval['wer_base_new'].mean() * 100
    avg_std = df_eval['wer_std_new'].mean() * 100
    avg_cust = df_eval['wer_cust_new'].mean() * 100

    print("\n" + "=" * 55)
    print(f"Final Evaluation Results (N = {len(df_eval)})")
    print("=" * 55)
    print(f"1. Whisper Baseline (Turbo):    {avg_base:.2f}%")
    print(f"2. Standard RAG (Baseline):     {avg_std:.2f}%")
    print(f"3. Error-Aware RAG (Proposed):  {avg_cust:.2f}%")
    print("=" * 55)

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Scored results successfully saved to '{output_file}'.")


if __name__ == "__main__":
    evaluate_pipeline()