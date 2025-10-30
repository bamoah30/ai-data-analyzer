"""
Test different Hugging Face models for the AI Data Analyzer
This script allows you to compare different models side by side
"""

from analyzer import get_insights
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

print("=" * 80)
print("SAMPLE PROMPT FOR ANALYSIS:")
print("=" * 80)
print(prompt)
print("\n")

# List of models to test
models_to_test = [
    ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral-7B (Recommended)"),
    ("HuggingFaceH4/zephyr-7b-beta", "Zephyr-7B"),
    ("Intel/neural-chat-7b-v3-1", "Neural-Chat-7B"),
]

print("=" * 80)
print("TESTING DIFFERENT MODELS")
print("=" * 80)
print("\n")

# Test single model (recommended approach for testing)
print("Testing Default Model (Mistral-7B - Recommended)")
print("-" * 80)

try:
    insights = get_insights(
        prompt,
        api_key=hf_token,
        provider="huggingface",
        model="mistralai/Mistral-7B-Instruct-v0.1"
    )
    
    print("Response:")
    print(insights)
    print("\n")
    
except Exception as e:
    print(f"Error: {e}\n")

