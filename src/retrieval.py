import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import diags


from src.normalization import SymmetricNormalizer


class ErrorAwareTFIDF:

    def __init__(self, correct_weight=0.1, error_weight=2.0):
        self.vectorizer = TfidfVectorizer()
        self.correct_weight = correct_weight
        self.error_weight = error_weight
        self.processor = SymmetricNormalizer()
        self.term_weights = None

    def fit_transform(self, df, whisper_col='whisper', ref_col='reference'):
        clean_whisper = df[whisper_col].apply(self.processor.clean)
        clean_ref = df[ref_col].apply(self.processor.clean)
        tfidf_matrix = self.vectorizer.fit_transform(clean_whisper)
        vocab = self.vectorizer.vocabulary_

        t_corr = {t: 0 for t in vocab}
        t_err = {t: 0 for t in vocab}
        for w, r in zip(clean_whisper, clean_ref):
            w_w = w.split()
            r_w = set(r.split())
            for word in w_w:
                if word in vocab:
                    if word in r_w:
                        t_corr[word] += 1
                    else:
                        t_err[word] += 1

        weights = np.ones(len(vocab))
        for term, idx in vocab.items():
            total = t_err[term] + t_corr[term]
            if total > 0:
                weights[idx] = self.correct_weight + (t_err[term] / total) * (self.error_weight - self.correct_weight)

        self.term_weights = weights
        return tfidf_matrix * diags(weights)

    def transform(self, texts):
        if isinstance(texts, str): texts = [texts]
        clean_texts = [self.processor.clean(t) for t in texts]
        return self.vectorizer.transform(clean_texts) * diags(self.term_weights)