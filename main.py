import os
import sys
import yaml
import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import joblib


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir(project_root)

from src.normalization import SymmetricNormalizer
from src.retrieval import ErrorAwareTFIDF
from src.llm_inference import call_llm


def main():
    print("\n" + "=" * 60)
    print("Starting Error-Aware RAG Inference Pipeline (FULL DATASET)")
    print("=" * 60)

    # 1. Load Configurations Safely
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'configs/config.yaml' not found. Please ensure it exists.")
        sys.exit(1)

    input_csv = config['paths']['input_csv']
    output_csv = config['paths']['output_csv']

    model_name = config['llm']['model_name']
    base_url = config['llm']['base_url']
    sys_prompt = config['llm']['system_prompt']
    llm_temp = config['llm'].get('temperature', 0.1)

    # 2. Initialize the API Client from YAML
    api_key = config['llm'].get('api_key')
    if not api_key:
        print("Error: API Key not found in config.yaml.")
        sys.exit(1)

    custom_http = httpx.Client(timeout=60.0)
    client = OpenAI(base_url=base_url, api_key=api_key, http_client=custom_http)

    # 3. Initialize Text Processor
    processor = SymmetricNormalizer()

    # 4. Load the Pre-Trained Artifacts
    std_artifact_path = "models/standard_tfidf_artifacts.joblib"
    custom_artifact_path = "models/error_aware_tfidf_artifacts.joblib"
    processed_text_path = "data/processed/fleurs_fa_rag_clean.csv"

    # Safety Check: Ensure the user built the database first
    if not (os.path.exists(std_artifact_path) and os.path.exists(custom_artifact_path)):
        print("Error: Pre-trained artifacts not found in the 'models/' folder.")
        print("Please run 'python scripts/build_vector_db.py' first to generate them.")
        sys.exit(1)

    print("Loading pre-trained vector matrices from disk...")
    std_artifacts = joblib.load(std_artifact_path)
    std_vectorizer = std_artifacts['vectorizer']
    std_matrix = std_artifacts['matrix']

    custom_artifacts = joblib.load(custom_artifact_path)
    custom_engine = custom_artifacts['engine']
    custom_matrix = custom_artifacts['matrix']

    print("Loading pre-cleaned Knowledge Base text...")
    rag_df = pd.read_csv(processed_text_path)

    # 5. Load the Evaluation Dataset (With Resume Capability)
    if os.path.exists(output_csv):
        print(f"Found existing progress. Loading from '{output_csv}' to resume...")
        df = pd.read_csv(output_csv)
    else:
        print(f"Loading fresh input data from '{input_csv}'...")
        df = pd.read_csv(input_csv)
        df['std_rag_output'] = None
        df['custom_rag_output'] = None

    print(f"FULL MODE: Processing {len(df)} rows.")
    print(f"Connecting to {model_name}...")

    # 6. Main Inference Loop
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_whisper = str(row['whisper_base'])
        clean_whisper = processor.clean(orig_whisper)

        # If the text is too short, just return the raw whisper output
        if len(clean_whisper) < 2:
            df.at[idx, 'std_rag_output'] = orig_whisper
            df.at[idx, 'custom_rag_output'] = orig_whisper
            continue

        # --- Baseline RAG ---
        if pd.isna(row['std_rag_output']):
            vec_std = std_vectorizer.transform([clean_whisper])
            sims_std = cosine_similarity(vec_std, std_matrix).flatten()
            best_std = sims_std.argsort()[-1]
            ctx_std = rag_df.iloc[best_std]['reference'] if sims_std[best_std] > 0.05 else ""

            out_std, success_std = call_llm(client, sys_prompt, ctx_std, orig_whisper, model_name, temperature=llm_temp)

            # Fallback to Whisper base if the API crashes or returns empty
            if success_std and out_std:
                df.at[idx, 'std_rag_output'] = processor.clean(out_std)
            else:
                df.at[idx, 'std_rag_output'] = clean_whisper

        # --- Error-Aware RAG ---
        if pd.isna(row['custom_rag_output']):
            vec_cust = custom_engine.transform([clean_whisper])
            sims_cust = cosine_similarity(vec_cust, custom_matrix).flatten()
            best_cust = sims_cust.argsort()[-1]
            ctx_cust = rag_df.iloc[best_cust]['reference'] if sims_cust[best_cust] > 0.05 else ""

            out_cust, success_cust = call_llm(client, sys_prompt, ctx_cust, orig_whisper, model_name,
                                              temperature=llm_temp)

            # Fallback to Whisper base if the API crashes or returns empty
            if success_cust and out_cust:
                df.at[idx, 'custom_rag_output'] = processor.clean(out_cust)
            else:
                df.at[idx, 'custom_rag_output'] = clean_whisper

        # Auto-Save progress every 10 rows
        if idx > 0 and idx % 10 == 0:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 7. Final Save
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 60)
    print(f"Full Pipeline finished! Results saved to '{output_csv}'.")
    print("=" * 60)


if __name__ == "__main__":
    main()