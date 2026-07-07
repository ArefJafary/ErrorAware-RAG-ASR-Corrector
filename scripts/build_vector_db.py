import os
import sys
import yaml
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.normalization import SymmetricNormalizer
from src.retrieval import ErrorAwareTFIDF


def build_database():
    print("\n" + "=" * 60)
    print("Building Error-Aware RAG Vector Database")
    print("=" * 60)

    # 1. Load Configurations
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    rag_db_path = config['paths']['rag_db_path']
    w_correct = config['retrieval']['correct_weight']
    w_error = config['retrieval']['error_weight']

    # 2. Initialize the Normalizer
    print("Initializing Symmetric Normalizer...")
    processor = SymmetricNormalizer()

    # 3. Load and Clean the Raw Knowledge Base
    print(f"Loading raw reference data from '{rag_db_path}'...")
    rag_df = pd.read_csv(rag_db_path).dropna()

    # Filter out very short segments that lack meaningful context
    rag_df = rag_df[rag_df['whisper'].str.split().str.len() > 2].reset_index(drop=True)

    print("Normalizing text (This might take a moment)...")
    # We save the cleaned text to a new column so we don't have to clean it repeatedly
    rag_df['whisper_clean'] = rag_df['whisper'].apply(processor.clean)
    rag_df['reference_clean'] = rag_df['reference'].apply(processor.clean)

    # 4. Train the Standard TF-IDF Matrix
    print("\nTraining Standard TF-IDF Matrix...")
    std_vectorizer = TfidfVectorizer()
    std_matrix = std_vectorizer.fit_transform(rag_df['whisper_clean'])

    # 5. Train the Error-Aware Penalty Matrix
    print(f"Training Error-Aware TF-IDF Matrix (W_c={w_correct}, W_e={w_error})...")
    custom_engine = ErrorAwareTFIDF(correct_weight=w_correct, error_weight=w_error)
    # The fit_transform inside your ErrorAwareTFIDF class already expects the dataframe
    custom_matrix = custom_engine.fit_transform(rag_df)

    # 6. Save the Artifacts
    os.makedirs("models", exist_ok=True)
    std_artifact_path = "models/standard_tfidf_artifacts.joblib"
    custom_artifact_path = "models/error_aware_tfidf_artifacts.joblib"

    print("\nSaving Vector Matrices to disk...")
    joblib.dump({'vectorizer': std_vectorizer, 'matrix': std_matrix}, std_artifact_path)
    joblib.dump({'engine': custom_engine, 'matrix': custom_matrix}, custom_artifact_path)

    processed_text_path = "data/processed/fleurs_fa_rag_clean.csv"
    os.makedirs("data/processed", exist_ok=True)
    rag_df.to_csv(processed_text_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("Database generation complete!")
    print(f"   - Standard Matrix: {std_artifact_path}")
    print(f"   - Error-Aware Matrix: {custom_artifact_path}")
    print(f"   - Cleaned RAG Text: {processed_text_path}")
    print("=" * 60)


if __name__ == "__main__":
    build_database()