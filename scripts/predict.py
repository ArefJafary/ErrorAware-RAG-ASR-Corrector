import os
import sys
import argparse
import yaml
import httpx
import pandas as pd
from openai import OpenAI
import joblib
from sklearn.metrics.pairwise import cosine_similarity


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.normalization import SymmetricNormalizer
from src.llm_inference import call_llm


def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Correct Whisper ASR errors using Error-Aware RAG.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The raw, erroneous Whisper ASR output in Persian."
    )
    args = parser.parse_args()
    user_input = args.text

    # 2. Load Configurations & API Key silently
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        sys.exit(1)

    model_name = config['llm']['model_name']
    base_url = config['llm']['base_url']
    sys_prompt = config['llm']['system_prompt']
    llm_temp = config['llm'].get('temperature', 0.1)
    api_key = config['llm'].get('api_key')
    if not api_key:
        print("Error: API Key not found in config.yaml.")
        sys.exit(1)

    # Initialize OpenAI Client
    custom_http = httpx.Client(timeout=60.0)
    client = OpenAI(base_url=base_url, api_key=api_key, http_client=custom_http)
    processor = SymmetricNormalizer()

    # 3. Load ONLY Error-Aware Pre-Trained Artifacts
    custom_artifact_path = "models/error_aware_tfidf_artifacts.joblib"
    processed_text_path = "data/processed/fleurs_fa_rag_clean.csv"

    if not os.path.exists(custom_artifact_path) or not os.path.exists(processed_text_path):
        print("Error: Pre-trained artifacts not found. Run 'python scripts/build_vector_db.py' first.")
        sys.exit(1)

    custom_artifacts = joblib.load(custom_artifact_path)
    custom_engine = custom_artifacts['engine']
    custom_matrix = custom_artifacts['matrix']
    rag_df = pd.read_csv(processed_text_path)

    # 4. Process the Input
    clean_whisper = processor.clean(user_input)

    # If the text is too short, just return the raw whisper output and exit cleanly
    if len(clean_whisper) < 2:
        print(user_input)
        sys.exit(0)

    # --- Error-Aware RAG Retrieval ---
    vec_cust = custom_engine.transform([clean_whisper])
    sims_cust = cosine_similarity(vec_cust, custom_matrix).flatten()
    best_cust = sims_cust.argsort()[-1]
    ctx_cust = rag_df.iloc[best_cust]['reference'] if sims_cust[best_cust] > 0.05 else ""

    # 5. Generate corrections via LLM
    out_cust, success = call_llm(client, sys_prompt, ctx_cust, user_input, model_name, temperature=llm_temp)

    # 6. Print ONLY the final output
    final_cust = processor.clean(out_cust if out_cust and success else user_input)
    print(final_cust)


if __name__ == "__main__":
    # Suppress basic warnings for a cleaner CLI experience
    import warnings

    warnings.filterwarnings("ignore")
    main()