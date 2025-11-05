from data_loader import load_file
from prompt_builder import build_prompt
from analyzer import get_insights
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Load your dataset
print("Loading data...")
df = load_file("sample_data/test.csv")
print(f"Loaded {len(df)} rows and {len(df.columns)} columns\n")

# Step 2: Build a prompt from the dataset
print("Building prompt...")
prompt = build_prompt(df)
print(f"Prompt generated ({len(prompt)} characters)\n")

# Step 3: Get insights from AI API
print("Fetching insights from AI API...")

# Choose your provider:
# provider = "openai"  # For OpenAI API
provider = "huggingface"  # For Hugging Face API (Free!)

api_key = os.getenv("HUGGINGFACE_API_KEY") if provider == "huggingface" else os.getenv("OPENAI_API_KEY")

if not api_key:
    print(f" Error: API key not found. Check your .env file for {provider.upper()}_API_KEY.")
    exit(1)

insights = get_insights(prompt, api_key=api_key, provider=provider)

# Step 4: Display the results
print("Insights received:\n")
print(insights)