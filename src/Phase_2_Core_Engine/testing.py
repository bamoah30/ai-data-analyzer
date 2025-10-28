from data_loader import load_file
from prompt_builder import build_prompt

df = load_file("sample_data/product_inventory.xlsx")
prompt = build_prompt(df)
print(prompt)