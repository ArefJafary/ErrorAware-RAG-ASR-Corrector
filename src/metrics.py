from jiwer import wer

def calculate_row_wer(reference, hypothesis):
    """Calculates the Word Error Rate between a ground truth and a prediction."""
    if not reference or len(reference.strip()) == 0:
        return 0.0
    try:
        return wer(reference, hypothesis)
    except Exception:
        return 1.0