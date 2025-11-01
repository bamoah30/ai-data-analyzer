import os
from transformers import pipeline

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

def get_response(prompt, format="text"):
    pipeline_model = pipeline("text-generation")

    # Set up the Hugging Face API token
    os.environ["HF_NAME_OR_EMAIL"] = "your_email@example.com"
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

    response = pipeline_model(prompt, max_length=100, temperature=0.5)

    if format == "json":
        return response
    elif format == "text":
        return response[0]["generated_text"]
    else:
        raise ValueError("Invalid response format")

if __name__ == "__main__":
    prompt = "How to integrate a model into a project?"
    format = "text"
    response = get_response(prompt, format)
    print(response)
```