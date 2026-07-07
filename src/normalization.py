import re
import hazm


class SymmetricNormalizer:
    """
    Used ONLY for Knowledge Base construction and Inference.
    Preserves enough morphological detail for the Error-Aware TF-IDF to work accurately.
    """

    def __init__(self):
        self.normalizer = hazm.Normalizer()

    def remove_consecutive_duplicates(self, text, max_repeats=2):
        if not text: return text
        words = text.split()
        result = []
        count = 0
        prev_word = None
        for word in words:
            if word == prev_word:
                count += 1
            else:
                count = 1
                prev_word = word
            if count <= max_repeats: result.append(word)
        return " ".join(result)

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = self.normalizer.normalize(text)
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = text.replace('\u200c', ' ')
        return self.remove_consecutive_duplicates(re.sub(r'\s+', ' ', text).strip())


class EvaluationNormalizer:
    """
    Used ONLY for the final WER calculation.
    Applies aggressive ITN and morphological stripping to ensure a perfectly fair evaluation.
    """

    def __init__(self):
        self.normalizer = hazm.Normalizer()

    def itn_persian(self, text):
        if not isinstance(text, str): return ""
        translation_table = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
        text = text.translate(translation_table)

        words = {
            'صفر': 0, 'یک': 1, 'دو': 2, 'سه': 3, 'چهار': 4, 'پنج': 5, 'شش': 6, 'هفت': 7, 'هشت': 8, 'نه': 9,
            'ده': 10, 'یازده': 11, 'دوازده': 12, 'سیزده': 13, 'چهارده': 14, 'پانزده': 15, 'شانزده': 16, 'هفده': 17,
            'هجده': 18, 'نوزده': 19,
            'بیست': 20, 'سی': 30, 'چهل': 40, 'پنجاه': 50, 'شصت': 60, 'هفتاد': 70, 'هشتاد': 80, 'نود': 90,
            'صد': 100, 'یکصد': 100, 'دویست': 200, 'سیصد': 300, 'چهارصد': 400, 'پانصد': 500, 'ششصد': 600, 'هفتصد': 700,
            'هشتصد': 800, 'نهصد': 900
        }
        magnitudes = {'هزار': 1000, 'میلیون': 1000000, 'میلیارد': 1000000000}

        tokens = text.split()
        result = []
        current_num = 0
        current_mag = 0
        in_number = False

        def commit():
            nonlocal current_num, current_mag, in_number, result
            if in_number:
                result.append(str(current_num + current_mag))
                current_num, current_mag, in_number = 0, 0, False

        i = 0
        while i < len(tokens):
            token = tokens[i]
            clean_token = token.replace('\u200c', '')

            if clean_token in words:
                in_number = True
                current_num += words[clean_token]
            elif clean_token in magnitudes:
                in_number = True
                if current_num == 0: current_num = 1
                current_mag += current_num * magnitudes[clean_token]
                current_num = 0
            elif clean_token == 'و' and in_number:
                next_token = tokens[i + 1].replace('\u200c', '') if i + 1 < len(tokens) else ""
                if next_token not in words and next_token not in magnitudes:
                    commit()
                    result.append(token)
            else:
                commit()
                result.append(token)
            i += 1

        commit()
        return ' '.join(result)

    def clean(self, text):
        if not isinstance(text, str): return ""

        text = self.normalizer.normalize(text)
        text = text.replace('\u200c', '')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = self.itn_persian(text)

        # 5. Surgical removals for fair WER
        text = re.sub(r'(^|\s)(می|نمی)\s+', r'\1\2', text)
        text = re.sub(r'\s+(تر|ترین)(?=\s|$)', r'\1', text)
        text = re.sub(r'\s+(ها|های|هایی)(?=\s|$)', r'\1', text)

        return re.sub(r'\s+', ' ', text).strip()