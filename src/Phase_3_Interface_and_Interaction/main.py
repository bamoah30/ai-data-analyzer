import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from core.data_loader import load_file
from core.prompt_builder import build_prompt
from core.analyzer import get_insights

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="AI Data Analyzer CLI and Hugging Face & OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file data.csv --provider huggingface
  python main.py --file data.xlsx --provider openai --max_tokens 1500
        """
    )
    
    parser.add_argument(
        '--file', 
        required=True, 
        help='Path to your data file (.csv, .xlsx, .json)'
    )
    parser.add_argument(
        '--provider', 
        choices=['huggingface', 'openai'], 
        default='huggingface',
        help='Which API provider to use (default: huggingface)'
    )
    parser.add_argument(
        '--model', 
        default=None,
        help='Model to use. Default: MiniMaxAI/MiniMax-M2:novita (HF) or gpt-3.5-turbo (OpenAI)'
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=1000, 
        help='Maximum tokens in the response (default: 1000)'
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7,
        help='Creativity level of response 0.0-1.0 (default: 0.7)'
    )
    
    args = parser.parse_args()

    # Validate file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return

    # Load the dataset
    print(f"Loading data from {args.file}...")
    try:
        df = load_file(args.file)
        print(f"Successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Build the prompt
    print("Building analysis prompt...")
    try:
        prompt = build_prompt(df)
        print(f"Prompt built successfully ({len(prompt)} characters)\n")
    except Exception as e:
        print(f"Error building prompt: {e}")
        return

    # Determine API key based on provider
    if args.provider == 'huggingface':
        env_key = "HUGGINGFACE_API_KEY"
    else:
        env_key = "OPENAI_API_KEY"
    
    api_key = os.getenv(env_key)
    
    if not api_key:
        print(f"Error: {env_key} not found in environment variables")
        print(f"  Set it with: export {env_key}=your-key-here")
        print(f"  Or create a .env file with: {env_key}=your-key-here")
        return

    # Get insights from selected provider
    print(f"Sending request to {args.provider} API...")
    print("(This may take 2-5 seconds)\n")
    try:
        insights = get_insights(
            prompt=prompt,
            api_key=api_key,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("=" * 80)
        print("AI-GENERATED INSIGHTS:")
        print("=" * 80)
        print(insights)
        print("=" * 80)
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error retrieving insights: {e}")
        return

if __name__ == "__main__":
    main()