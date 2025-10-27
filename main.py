# import openai
# import pandas as pd

# # Load your CSV
# def load_csv(file_path):
#     return pd.read_csv(file_path)

# # Generate prompt from data
# def generate_prompt(df):
#     return f"Analyze this dataset:\n{df.head().to_string()}"

# # Call OpenAI API
# def analyze_data(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response['choices'][0]['message']['content']

# # Example usage
# if __name__ == "__main__":
#     df = load_csv("your_data.csv")
#     prompt = generate_prompt(df)
#     insights = analyze_data(prompt)
#     print(insights)
