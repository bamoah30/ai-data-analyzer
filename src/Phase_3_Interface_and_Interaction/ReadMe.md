# 🧠 Phase 3 – Interface and Interaction

> 📌 This phase demonstrates Bernard’s ability to build user-facing interfaces around a modular AI engine powered by Hugging Face. It focuses on accessibility, usability, and real-world deployment readiness.

> ✅ Status: Phase 3 in progress (as of Week 6 of my journey).

---

## 🎯 Phase Objectives

- Build a **Command-Line Interface (CLI)** using `argparse`
- Build a **Graphical User Interface (GUI)** using `streamlit`
- Integrate both interfaces with the Hugging Face API
- Maintain modularity, testability, and clean separation of concerns

---

## 🧩 Interface Modules

| File/Folder            | Purpose                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- |
| `main.py`              | CLI entry point using `argparse`                                                |
| `app/`                 | Streamlit GUI folder                                                            |
| `app/streamlit_app.py` | Streamlit interface for file upload and insight display                         |
| `core/`                | Contains Phase 2 modules (`data_loader.py`, `prompt_builder.py`, `analyzer.py`) |
| `utils/`               | Shared utilities (e.g., file validation, error handling)                        |

---

## 🖥️ CLI Overview – `main.py`

### ✅ Features

- Accepts file path as argument
- Optional model selection (`--model bert`, `--model distilbert`)
- Displays insights in terminal

### 🧪 Example Usage

```bash
python main.py --file sample_data/test.csv --model distilbert
```

### 🧠 Argparse Setup

```python
import argparse

parser = argparse.ArgumentParser(description="AI Data Analyzer CLI")
parser.add_argument('--file', required=True, help='Path to data file')
parser.add_argument('--model', default='distilbert', help='Hugging Face model to use')
args = parser.parse_args()
```

---

## 🌐 GUI Overview – `streamlit_app.py`

### ✅ Features

- File uploader (CSV, Excel, JSON)
- Model selector dropdown (e.g., `bert-base-uncased`, `distilbert-base-uncased`)
- Insight display in scrollable layout
- Optional: download insights as `.txt` or `.md`

### 🧠 Streamlit App Skeleton

```python
import streamlit as st
from core.data_loader import load_file
from core.prompt_builder import build_prompt
from core.analyzer import get_insights

st.title("AI Data Analyzer")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])
model = st.selectbox("Choose a Hugging Face model", ["bert-base-uncased", "distilbert-base-uncased"])

if uploaded_file:
    df = load_file(uploaded_file)
    prompt = build_prompt(df)
    insights = get_insights(prompt, model=model)
    st.write(insights)
```

---

## 🔐 API Key Setup (Hugging Face)

To use the Hugging Face Inference API, you’ll need a valid API token.

### ✅ Option 1: Environment Variable (Recommended for CLI)

```bash
export HF_API_TOKEN=your-token-here
```

### 🧪 Option 2: Use a `.env` File (with `python-dotenv`)

```env
HF_API_TOKEN=your-token-here
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_API_TOKEN")
```

✅ Add this to `.gitignore`:

```text
.env
```

---

## 🧪 Testing Strategy

| Interface | Test Type                      | Example                                         |
| --------- | ------------------------------ | ----------------------------------------------- |
| CLI       | Manual + Unit                  | Run with sample files, assert output            |
| GUI       | Manual + Hot Reload            | Upload edge-case files, test layout             |
| Shared    | Unit tests for `core/` modules | Validate prompt structure, API response parsing |

---

## 🧠 Strategic Intent

This phase demonstrates Bernard’s ability to:

- Build accessible tools for both technical and non-technical users
- Integrate Hugging Face models into real-world interfaces
- Prepare the project for public demos, deployment, and user feedback

---

> 🧭 Phase Summary: Bernard is transforming a modular AI engine into a user-facing product powered by Hugging Face. By integrating CLI and GUI interfaces, he’s making the tool accessible, testable, and ready for real-world use.

> 🗓️ Milestone: CLI and GUI scaffolding underway. Hugging Face integration in progress. Public demo planned for Week 7.
