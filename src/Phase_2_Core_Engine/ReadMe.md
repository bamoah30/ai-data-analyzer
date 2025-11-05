# Phase 2 – Core Engine

# Project: AI Data Analyzer

> This phase demonstrates Bernard's ability to build modular AI pipelines using Python, OpenAI API (or Hugging Face API), and pandas. It lays the foundation for CLI and GUI integration in Phase 3.

> Status: Phase 2 in progress ( expected completion by end of week 6).

This phase builds the technical backbone of the **AI Data Analyzer**. It includes the core logic for ingesting data, generating prompts, and retrieving insights from an AI API (OpenAI or Hugging Face). Each module is designed to be modular, testable, and extensible—laying the groundwork for CLI and GUI integration in later phases.

---

## Modules in This Phase

| File                | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| `data_loader.py`    | Loads CSV, Excel, and JSON files into pandas DataFrames         |
| `prompt_builder.py` | Converts DataFrame summaries into structured prompts for AI API |
| `analyzer.py`       | Sends prompts to AI API and returns insights                    |
| `sample_data/`      | Contains test files for development and debugging               |

---

## Module Details

### `data_loader.py`

Loads data files and converts them into pandas DataFrames for analysis.

**Supported Formats:**

- `.csv` — Comma-separated values (default encoding: UTF-8)
- `.xlsx` — Excel workbooks (single sheet)
- `.json` — JSON arrays or objects

**Behavior:**

- Returns a pandas DataFrame
- Raises `ValueError` for unsupported formats or corrupted files
- Automatically handles common encoding issues (UTF-8, Latin-1)

**Example Usage:**

```python
from data_loader import load_file

df = load_file("sample_data/sales_data.csv")
print(df.head())  # View first 5 rows
print(df.info())  # View column types and null counts
```

### `prompt_builder.py`

Converts DataFrame metadata into structured prompts for AI API analysis.

**What It Does:**

- Generates statistical summaries using `df.describe()`
- Extracts column names and data types
- Counts missing values per column
- Formats all metadata into a readable, structured prompt

**Prompt Structure Example:**

```
Analyze the following dataset summary:

Dataset Shape: 500 rows, 5 columns

Column Information:
- Age (int64): Mean=42.3, Std=15.2, Min=18, Max=78
- Salary (float64): Mean=65000.0, Std=20000.0, Min=25000, Max=150000
- Department (object): 4 unique values, 2 missing values
- Experience (int64): Mean=8.5, Std=6.1, Min=0, Max=35
- Bonus (float64): Mean=5000.0, Std=3000.0, Min=0, Max=25000

Missing Values: 2 (0.4% of total data)

Please provide insights on trends, correlations, and data quality.
```

**Example Usage:**

```python
from prompt_builder import build_prompt

prompt = build_prompt(df)
print(prompt)  # View the generated prompt
```

### `analyzer.py`

Sends structured prompts to an AI API and retrieves AI-generated insights. Supports both OpenAI and Hugging Face APIs.

**Configuration:**

- **OpenAI:** Uses `gpt-3.5-turbo` model (configurable)
- **Hugging Face:** Uses `MiniMaxAI/MiniMax-M2:novita` or similar models
- Requires a valid API key
- Typical response time: 2-5 seconds
- Returns insights as a formatted string

**Example Usage:**

```python
from analyzer import get_insights
import os

# For OpenAI
api_key = os.getenv("OPENAI_API_KEY")
insights = get_insights(prompt, api_key=api_key, provider="openai")

# For Hugging Face
hf_token = os.getenv("HUGGINGFACE_API_KEY")
insights = get_insights(prompt, api_key=hf_token, provider="huggingface")
```

---

## Dependencies

Install all required packages with:

```bash
pip install pandas openai requests python-dotenv
```

**Version Requirements:**

- `pandas >= 1.3.0`
- `openai >= 1.0.0` (supports OpenAI API v1)
- `requests >= 2.25.0` (for Hugging Face API)
- `python-dotenv >= 0.19.0`

### Optional Dependencies (for Future Phases)

These packages are not required in Phase 2 but will be useful as the project evolves:

| Package         | Purpose                                                                                |
| --------------- | -------------------------------------------------------------------------------------- |
| `python-dotenv` | Load your API keys from a `.env` file instead of hardcoding them                       |
| `argparse`      | Build a command-line interface (CLI) for Phase 3 (standard library, no install needed) |
| `streamlit`     | Create a browser-based GUI for non-technical users in Phase 3                          |

Install them with:

```bash
pip install python-dotenv streamlit
```

---

## API Key Setup

To use the AI analysis features, you can choose between **OpenAI** or **Hugging Face** APIs. Both can be set up in similar ways.

### Quick Choice Guide

| Feature           | OpenAI                          | Hugging Face              |
| ----------------- | ------------------------------- | ------------------------- |
| **Free Tier**     | Limited ($5 trial)              | Free & Unlimited          |
| **Card Required** | Credit/Debit card               | No card needed            |
| **Setup Time**    | 5 minutes                       | 2 minutes                 |
| **Model Quality** | Excellent (gpt-3.5-turbo/gpt-4) | Good (MiniMax-M2, others) |
| **Best For**      | Production, high accuracy       | Development, free testing |

---

## OpenAI API Setup (Optional)

### Option 1: Environment Variable (Recommended for CLI)

Set your API key in your terminal session:

```bash
export OPENAI_API_KEY=your-key-here
```

Then access it in your Python code:

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

### Option 2: Use a `.env` File (Recommended for Development)

This method is ideal for local development and prevents accidental API key exposure.

**Step 1:** Create a `.env` file in your project root

```env
OPENAI_API_KEY=your-key-here
```

**Step 2:** Load the key in your Python script

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**Step 3:** Install `python-dotenv` (if not already installed)

```bash
pip install python-dotenv
```

**Step 4:** Add `.env` to `.gitignore` to prevent accidental commits

```text
.env
```

**Get Your OpenAI API Key:**

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in with your account
3. Create a new API key
4. Copy and paste it into your `.env` file or environment variable

---

## Hugging Face API Setup (Recommended for Free Users)

### Option 1: Environment Variable (Recommended for CLI)

Set your API key in your terminal session:

```bash
export HUGGINGFACE_API_KEY=your-token-here
```

Then access it in your Python code:

```python
import os
hf_token = os.getenv("HUGGINGFACE_API_KEY")
```

### Option 2: Use a `.env` File (Recommended for Development)

This method is ideal for local development and prevents accidental API key exposure.

**Step 1:** Create a `.env` file in your project root

```env
HUGGINGFACE_API_KEY=your-token-here
```

**Step 2:** Load the token in your Python script

```python
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
```

**Step 3:** Install `python-dotenv` (if not already installed)

```bash
pip install python-dotenv
```

**Step 4:** Add `.env` to `.gitignore` to prevent accidental commits

```text
.env
```

**Get Your Hugging Face API Token:**

1. Visit [Hugging Face](https://huggingface.co/join) to create a free account
2. **No credit card required**
3. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token" and select "Read" access type
5. Copy your token and paste it into your `.env` file or environment variable

**Tip:** Hugging Face is ideal for development and testing because it's completely free with no payment method required!

---

## How to Run the Core Engine

### Step 1: Prepare Your Environment

```bash
# Clone or navigate to your project directory
cd ai-data-analyzer

# Install dependencies
pip install pandas openai requests python-dotenv

# Create and configure your .env file
# For Hugging Face (Recommended):
echo "HUGGINGFACE_API_KEY=your-token-here" > .env

# OR for OpenAI:
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Step 2: Prepare Sample Data

Place a test file in the `sample_data/` folder:

```text
ai-data-analyzer/
└── Phase 2 - Core Engine/
    ├── data_loader.py
    ├── prompt_builder.py
    ├── analyzer.py
    ├── .env
    └── sample_data/
        └── test.csv
```

**Sample CSV format:**

```csv
Age,Salary,Department,Experience,Bonus
28,45000,Sales,2,2000
35,62000,Engineering,7,5000
42,75000,Marketing,12,4500
```

### Step 3: Create a Test Script

Create a file named `test_pipeline.py`:

```python
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
```

### Step 4: Run the Test

```bash
python test_pipeline.py
```

### Step 5: Validate the Output

You should see output like:

```text
Loading data...
Loaded 3 rows and 5 columns

Building prompt...
Prompt generated (1245 characters)

Fetching insights from AI API...
Insights received:

Top 3 Insights:
1. The dataset shows a positive correlation between Experience and Salary—employees with more years gain approximately 2.5% salary increase per year.
2. Department distribution is balanced across Sales and Engineering, with Marketing underrepresented (33% vs 50%).
3. Bonus structure is tied to department—Engineering receives higher bonuses (avg $5000) compared to Sales (avg $2000).

Data Quality Notes:
- No missing values detected
- All salary values fall within realistic ranges
- Experience values are consistent with age distribution
```

This confirms:

- Data was successfully loaded
- Prompt was correctly generated
- AI API returned meaningful insights

---

## Troubleshooting

### API Key Issues

**Error: `"Incorrect API key provided"` (OpenAI)**

- Verify your API key is correct at [OpenAI Platform](https://platform.openai.com/api-keys)
- Check that your `.env` file is in the project root
- Ensure no extra spaces or quotes in the API key

**Error: `"Invalid token"` (Hugging Face)**

- Verify your token is correct at [Hugging Face Tokens](https://huggingface.co/settings/tokens)
- Check that your `.env` file is in the project root
- Ensure the token has "Read" access enabled

**Error: `"Rate limit exceeded"`**

- OpenAI API has usage limits (default: 3 requests/min for free tier)
- Hugging Face API has rate limits (higher for free tier)
- Wait a few minutes before retrying

### File Loading Issues

**Error: `"No such file or directory: sample_data/test.csv"`**

- Verify the file exists in the correct location
- Use absolute paths if relative paths don't work:
  ```python
  df = load_file("/absolute/path/to/test.csv")
  ```

**Error: `"Unsupported file format"`**

- Check that your file extension is `.csv`, `.xlsx`, or `.json`
- Verify the file is not corrupted (try opening it in Excel or a text editor)

**Error: `"UnicodeDecodeError"`**

- Your file may use a different encoding (e.g., Latin-1, UTF-16)
- Try converting the file to UTF-8 before loading

### DataFrame Issues

**Warning: `"Empty DataFrame returned"`**

- Verify your file contains actual data rows (not just headers)
- Check for extra blank rows at the beginning or end

**Error: `"DataFrame has NaN values in all columns"`**

- Ensure your column names are recognized by the data loader
- Verify your file uses standard delimiters (comma, tab, etc.)

---

## What's Next – Phase 3: Interface and Interaction

In the next phase, Bernard will integrate the core engine into user-facing interfaces that make the tool more accessible and interactive.

### Planned Interfaces

| Interface Type            | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| CLI (Command Line)        | Use `argparse` to allow users to run the analyzer via terminal commands   |
| GUI (Graphical Interface) | Use Streamlit to create a browser-based interface for non-technical users |

### Integration Goals

- Connect `data_loader.py`, `prompt_builder.py`, and `analyzer.py` into a unified pipeline
- Allow users to upload files and receive insights without touching the code
- Maintain modularity so both interfaces use the same backend logic

### Strategic Intent

This phase will demonstrate Bernard's ability to:

- Build user-friendly tools on top of technical foundations
- Design for accessibility and real-world use
- Prepare the project for deployment and public engagement

---

## Cost Considerations

### OpenAI API

The OpenAI API is **pay-as-you-go**. Typical costs for Phase 2:

- **gpt-3.5-turbo:** ~$0.001-0.002 per request (50-100 rows of data)
- **gpt-4:** ~$0.03 per request (more expensive but higher quality)

**Estimate:** Testing this phase with 100 requests = ~$0.10-0.30

Monitor your usage at [OpenAI Usage Dashboard](https://platform.openai.com/usage/overview).

### Hugging Face API

The Hugging Face API is **completely free** with generous rate limits:

- **No payment required**
- **No credit card needed**
- **Unlimited requests** (within fair use policy)
- **Multiple free models** available (Mistral-7B, Zephyr, and more)

**Estimate:** Testing this phase with unlimited requests = **$0.00**

Monitor your usage at [Hugging Face API Dashboard](https://huggingface.co/settings/tokens).

---

## Testing Checklist

Before moving to Phase 3, verify:

- [ ] All three modules (`data_loader.py`, `prompt_builder.py`, `analyzer.py`) import without errors
- [ ] `load_file()` successfully loads `.csv`, `.xlsx`, and `.json` files
- [ ] `build_prompt()` generates readable prompts from DataFrames
- [ ] `get_insights()` returns insights from your chosen API within 10 seconds
- [ ] Error handling works for invalid file formats and missing API keys
- [ ] `.env` file is properly added to `.gitignore`
- [ ] API keys are securely stored in `.env` file (not hardcoded)

---

> **Phase Summary:** Bernard has built a modular AI engine using Python, pandas, and flexible API support (OpenAI or Hugging Face), with clean architecture and comprehensive documentation. This phase sets the foundation for user-facing interfaces in Phase 3.

> **Milestone:** Core engine modules implemented and tested. CLI and GUI scaffolding begins in Phase 3.
