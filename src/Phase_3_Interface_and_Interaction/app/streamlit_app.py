"""
streamlit_app.py - Graphical User Interface for AI Data Analyzer

This module provides a web-based interface using Streamlit for uploading datasets,
selecting models, and displaying AI-generated insights. It's designed to be accessible
for both technical and non-technical users.

Key features:
    - File uploader (CSV, Excel, JSON)
    - Model and provider selector
    - Temperature and max tokens configuration
    - Real-time insight generation with loading state
    - Download insights as text or markdown
    - Responsive layout with proper error handling
"""

import sys
import os
from pathlib import Path

# MUST be first - add parent directory to path BEFORE any other imports
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Import core modules
from core.data_loader import load_file
from core.prompt_builder import build_prompt, build_prompt_with_sample
from core.analyzer import get_insights, estimate_cost

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "insights" not in st.session_state:
    st.session_state.insights = None

if "prompt_generated" not in st.session_state:
    st.session_state.prompt_generated = False

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Provider selection
    provider = st.radio(
        "Select API Provider:",
        options=["huggingface", "openai"],
        help="Choose between free Hugging Face or paid OpenAI"
    )
    
    # Model selection based on provider
    if provider == "huggingface":
        model = "MiniMaxAI/MiniMax-M2:novita"
        st.info("📌 Using Novita (MiniMax-M2) model for analysis")
    else:
        model_options = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview",
        ]
        model = st.selectbox(
            "Select Model:",
            options=model_options,
            help="Choose an OpenAI model for analysis"
        )
    
    st.divider()
    
    # Advanced parameters
    st.markdown("### 🎛️ Advanced Parameters")
    
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum length of the AI response"
    )
    
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    include_sample = st.checkbox(
        "Include sample data in analysis",
        value=True,
        help="Add first 5 rows to the analysis prompt"
    )
    
    st.divider()
    
    # API key management
    st.markdown("### 🔑 API Key")
    
    env_var = "HUGGINGFACE_API_KEY" if provider == "huggingface" else "OPENAI_API_KEY"
    api_key_from_env = os.getenv(env_var)
    
    if api_key_from_env:
        st.success(f"✓ {provider.title()} API key loaded from environment")
    else:
        st.warning(f"⚠️ No {provider.title()} API key in environment")
    
    api_key_input = st.text_input(
        f"Enter {provider.title()} API Key:",
        type="password",
        help=f"Provide your {provider.title()} API key here"
    )
    
    if api_key_input:
        api_key_from_env = api_key_input
    
    if not api_key_from_env:
        st.error(f"❌ API key is required to proceed")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="main-header">🧠 AI Data Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload your data and get AI-powered insights instantly</div>',
    unsafe_allow_html=True
)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# ============================================================================
# LEFT COLUMN - FILE UPLOAD AND PREVIEW
# ============================================================================

with col1:
    st.markdown("### 📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel (.xlsx), JSON"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location for processing
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the dataframe
            df = load_file(temp_path)
            st.session_state.uploaded_df = df
            
            # Display success message
            st.markdown(
                '<div class="success-box">✓ File uploaded successfully</div>',
                unsafe_allow_html=True
            )
            
            # Display dataset info
            st.markdown("#### 📊 Dataset Overview")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Rows", df.shape[0])
            with col_info2:
                st.metric("Columns", df.shape[1])
            with col_info3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Display data preview
            with st.expander("👀 View Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Display column information
            with st.expander("📋 Column Information", expanded=False):
                col_info_text = "| Column | Type | Non-Null | Unique |\n"
                col_info_text += "|--------|------|----------|--------|\n"
                
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    non_null = df[col].notna().sum()
                    unique = df[col].nunique()
                    col_info_text += f"| {col} | {col_type} | {non_null} | {unique} |\n"
                
                st.markdown(col_info_text)
            
            # Clean up temp file
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.uploaded_df = None

# ============================================================================
# RIGHT COLUMN - ANALYSIS AND INSIGHTS
# ============================================================================

with col2:
    if st.session_state.uploaded_df is not None:
        st.markdown("### 📈 Analysis & Insights")
        
        # Check if API key is available
        if not api_key_from_env:
            st.error(
                f"❌ API key required\n\n"
                f"Please provide your {provider.title()} API key in the sidebar "
                f"or set the {env_var} environment variable."
            )
        else:
            # Generate prompt button
            if st.button(
                "🚀 Generate Insights",
                use_container_width=True,
                type="primary"
            ):
                try:
                    with st.spinner("📄 Generating insights... This may take 2-5 seconds"):
                        # Build prompt
                        if include_sample:
                            prompt = build_prompt_with_sample(
                                st.session_state.uploaded_df,
                                sample_rows=5
                            )
                        else:
                            prompt = build_prompt(st.session_state.uploaded_df)
                        
                        # Get insights from API
                        insights = get_insights(
                            prompt=prompt,
                            api_key=api_key_from_env,
                            provider=provider,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        st.session_state.insights = insights
                        st.session_state.prompt_generated = True
                        
                except Exception as e:
                    st.error(f"❌ Error generating insights: {str(e)}")
            
            # Display insights if available
            if st.session_state.insights:
                st.markdown("---")
                st.markdown("#### 📈 AI-Generated Insights")
                
                # Display insights in a nice format
                st.markdown(st.session_state.insights)
                
                # Download options
                st.markdown("---")
                st.markdown("#### 💾 Download Insights")
                
                col_txt, col_md = st.columns(2)
                
                with col_txt:
                    txt_content = st.session_state.insights
                    st.download_button(
                        label="📄 Download as TXT",
                        data=txt_content,
                        file_name=f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_md:
                    md_content = f"# Data Analysis Insights\n\n{st.session_state.insights}"
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=md_content,
                        file_name=f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                # Cost estimation
                with st.expander("💰 Cost Estimation", expanded=False):
                    if include_sample:
                        prompt = build_prompt_with_sample(
                            st.session_state.uploaded_df,
                            sample_rows=5
                        )
                    else:
                        prompt = build_prompt(st.session_state.uploaded_df)
                    
                    cost_info = estimate_cost(prompt, provider=provider, model=model)
                    
                    if "error" not in cost_info:
                        col_est1, col_est2, col_est3 = st.columns(3)
                        
                        with col_est1:
                            st.metric("Input Tokens", cost_info["estimated_input_tokens"])
                        with col_est2:
                            st.metric("Output Tokens", cost_info["estimated_output_tokens"])
                        with col_est3:
                            st.metric("Est. Cost", f"${cost_info['estimated_cost_usd']}")
                        
                        st.info(cost_info.get("note", ""))
    else:
        st.info(
            "👈 Start by uploading a dataset using the file uploader on the left\n\n"
            "Supported formats: CSV, Excel (.xlsx), JSON"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <p>🧠 AI Data Analyzer | Powered by Hugging Face & OpenAI</p>
        <p>Phase 3 - Interface and Interaction</p>
    </div>
    """,
    unsafe_allow_html=True
)