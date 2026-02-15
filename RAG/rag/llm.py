from openai import OpenAI

def chat_completion(api_key: str, model: str, system: str, user: str) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
