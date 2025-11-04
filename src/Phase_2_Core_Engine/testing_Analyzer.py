"""
Test different Hugging Face models for the AI Data Analyzer
This script allows you to compare different models side by side
"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Hugging Face API token
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if not hf_token:
    print("Error: HUGGINGFACE_API_KEY not found in environment variables")
    print("Please set your token in .env file or as an environment variable")
    exit(1)

# Sample data analysis prompt
prompt = """Analyze the following dataset summary:

Dataset Shape: 100 rows, 6 columns

Column Information:
- Age (int64): Mean=35.4, Std=7.2, Min=26, Max=50
- Salary (int64): Mean=63250, Std=16340, Min=43000, Max=95000
- Experience (int64): Mean=7.8, Std=5.6, Min=1, Max=18
- Department (object): 4 unique values
- Bonus (int64): Mean=4590, Std=2140, Min=1800, Max=9000
- Performance_Score (int64): Mean=86.2, Std=4.8, Min=78, Max=95

Missing Values: None detected

Please provide insights on trends, correlations, and data quality."""

# print("=" * 80)
# print("SAMPLE PROMPT FOR ANALYSIS:")
# print("=" * 80)
# print(prompt)
# print("\n")

# # List of models to test
# models_to_test = [
#     ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral-7B (Recommended)"),
#     ("HuggingFaceH4/zephyr-7b-beta", "Zephyr-7B"),
#     ("Intel/neural-chat-7b-v3-1", "Neural-Chat-7B"),
# ]

# print("=" * 80)
# print("TESTING DIFFERENT MODELS")
# print("=" * 80)
# print("\n")

# # Test single model (recommended approach for testing)
# print("Testing Default Model (Mistral-7B - Recommended)")
# print("-" * 80)

# try:
#     insights = get_insights(
#         prompt,
#         api_key=hf_token,
#         provider="huggingface",
#         model="mistralai/Mistral-7B-Instruct-v0.1"
#     )
    
#     print("Response:")
#     print(insights)
#     print("\n")
    
# except Exception as e:
#     print(f"Error: {e}\n")



# from huggingface_hub import InferenceClient


# from dotenv import load_dotenv
# import os

# load_dotenv()
# token = os.getenv("HUGGINGFACE_API_KEY")

# client = InferenceClient(model="...", token=token)



# def ask_model(prompt: str) -> str:
#     client = InferenceClient(model="https://api-inference.huggingface.co/models/google/flan-t5-xxl", token= token)
#     return client.text_generation(prompt)

# response = ask_model(prompt)
# print("Response:")
# print(response)
import os
import requests
import json
from typing import Optional

API_URL = "https://router.huggingface.co/v1/chat/completions"
# prefer previously-loaded hf_token if available; fall back to HF_TOKEN env var
hf_token_env = globals().get("hf_token") or os.getenv("HF_TOKEN")
if not hf_token_env:
    print("Warning: no Hugging Face token found in `hf_token` or HF_TOKEN environment variable")

headers = {
    "Authorization": f"Bearer {hf_token_env}" if hf_token_env else "",
}


def _extract_text_from_choice(choice: dict) -> Optional[str]:
    # message can be a string or a dict with different shapes
    if not isinstance(choice, dict):
        return None

    # common chat-completion shapes
    msg = choice.get("message")
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            # look for parts (Hugging Face often uses {'parts': [...]})
            parts = content.get("parts") or content.get("text") or content.get("segments")
            if isinstance(parts, list):
                return "".join(str(p) for p in parts)
            if isinstance(parts, str):
                return parts
            # try text key
            text = content.get("text")
            if isinstance(text, str):
                return text

    # older completion format
    text = choice.get("text")
    if isinstance(text, str):
        return text

    return None


def query(payload: dict, timeout: int = 30) -> str:
    """Send payload to the Hugging Face chat completions endpoint and
    return a plain text string. This function is tolerant to several
    response shapes and will return the raw text when possible.

    Raises RuntimeError on network errors or non-2xx responses.
    """
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Request to {API_URL} failed: {e}") from e

    # If response is not JSON, return raw text
    try:
        data = resp.json()
    except ValueError:
        return resp.text

    # Try several common shapes for chat/generation responses
    if isinstance(data, dict):
        # choices -> message -> content/parts
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            text = _extract_text_from_choice(first)
            if text:
                return text

        # top-level generated_text
        gen = data.get("generated_text")
        if isinstance(gen, str):
            return gen

        # common other keys
        for key in ("output", "response", "result"):
            val = data.get(key)
            if isinstance(val, str):
                return val

    # fallback: return compact JSON string
    return json.dumps(data)


# Build payload and request textual response
if __name__ == "__main__":
    response_text = query({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "MiniMaxAI/MiniMax-M2:novita"
    })

    print(response_text)


