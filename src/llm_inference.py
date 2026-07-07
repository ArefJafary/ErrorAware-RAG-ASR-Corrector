import time

def call_llm(client, system_prompt, context, user_text, model_name, temperature=0.1, max_retries=3):
    user_prompt = f"دیتای RAG:\n{context}\n\nمتن ورودی:\n{user_text}"

    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )
            return res.choices[0].message.content.strip(), True

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"\n⏳ Rate limit hit (429). Retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"\n⚠️ API Error: {error_msg}")
                return "", False

    return "", False