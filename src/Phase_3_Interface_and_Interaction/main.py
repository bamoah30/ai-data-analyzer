import argparse
import os
from dotenv import load_dotenv
from core.data_loader import load_file
from core.prompt_builder import build_prompt
from core.analyzer import get_insights

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="AI Data Analyzer CLI – Hugging Face & OpenAI")
    parser.add_argument('--file', required=True, help='Path to your data file (.csv, .xlsx, .json)')
    parser.add_argument('--model', default='MiniMaxAI/MiniMax-M2:novita', help='Model to use (Hugging Face or OpenAI)')
    parser.add_argument('--provider', choices=['huggingface', 'openai'], default='huggingface', help='Which API to use')
    parser.add_argument('--hf_token', default=os.getenv("HF_API_TOKEN"), help='Hugging Face API token')
    parser.add_argument('--openai_key', default=os.getenv("OPENAI_API_KEY"), help='OpenAI API key')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Maximum tokens in the response')
    parser.add_argument('--temperature', type=float, default=0.7, help='Creativity level of the response')
    args = parser.parse_args()

    # Load the dataset
    try:
        df = load_file(args.file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Build the prompt
    try:
        prompt = build_prompt(df)
    except Exception as e:
        print(f"Error building prompt: {e}")
        return

    # Get insights from selected provider
    try:
        api_key = args.hf_token if args.provider == 'huggingface' else args.openai_key
        insights = get_insights(
            prompt=prompt,
            api_key=api_key,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print("\nInsights:")
        print(insights)
    except Exception as e:
        print(f"Error retrieving insights: {e}")

if __name__ == "__main__":
    main()
