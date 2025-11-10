"""
prompt_builder.py - Convert DataFrame metadata into structured prompts for OpenAI

This module generates comprehensive prompts from pandas DataFrames by extracting
statistical summaries, column information, and data quality metrics. The prompts
are formatted to provide optimal context for AI-powered data analysis.

Key features:
- Statistical summaries using df.describe()
- Column names and data types extraction
- Missing value detection and reporting
- Structured, readable prompt formatting
"""

import pandas as pd
import numpy as np


def build_prompt(df):
    """
    Build a structured prompt from a pandas DataFrame for OpenAI analysis.
    
    This function extracts comprehensive metadata from the DataFrame including:
    - Dataset shape (rows and columns)
    - Column information (names, types, statistics)
    - Missing value analysis
    - Statistical summaries for numerical columns
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
        
    Returns:
        str: A formatted prompt string ready for OpenAI API
        
    Raises:
        ValueError: If the DataFrame is empty or invalid
    """
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty - cannot build prompt from empty data")
    
    # Initialize prompt components
    prompt_parts = []
    
    # Header
    prompt_parts.append("Analyze the following dataset summary:")
    prompt_parts.append("")
    
    # Dataset shape
    rows, cols = df.shape
    prompt_parts.append(f"Dataset Shape: {rows} rows, {cols} columns")
    prompt_parts.append("")
    
    # Column Information
    prompt_parts.append("Column Information:")
    
    for col in df.columns:
        col_info = _build_column_info(df, col)
        prompt_parts.append(col_info)
    
    prompt_parts.append("")
    
    # Missing values summary
    missing_info = _build_missing_values_info(df)
    prompt_parts.append(missing_info)
    prompt_parts.append("")
    
    # Request for insights
    prompt_parts.append("Please provide insights on trends, correlations, and data quality.")
    
    # Join all parts into a single prompt
    prompt = "\n".join(prompt_parts)
    
    return prompt


def _build_column_info(df, column_name):
    """
    Build detailed information string for a single column.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the column
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Formatted string with column statistics
    """
    col_data = df[column_name]
    dtype = str(col_data.dtype)
    
    # Count missing values
    missing_count = col_data.isna().sum()
    
    # Start building the info string
    info_parts = [f"- {column_name} ({dtype})"]
    
    # For numeric columns, add statistical summary
    if pd.api.types.is_numeric_dtype(col_data):
        # Get statistics, handling potential issues with all-NaN columns
        if not col_data.isna().all():
            mean_val = col_data.mean()
            std_val = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            
            # Format numbers appropriately
            if dtype in ['int64', 'int32']:
                stats = f"Mean={mean_val:.1f}, Std={std_val:.1f}, Min={int(min_val)}, Max={int(max_val)}"
            else:
                stats = f"Mean={mean_val:.1f}, Std={std_val:.1f}, Min={min_val:.1f}, Max={max_val:.1f}"
            
            info_parts.append(f": {stats}")
    
    # For categorical/object columns, add unique value count
    elif pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
        unique_count = col_data.nunique()
        info_parts.append(f": {unique_count} unique values")
    
    # For datetime columns
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        if not col_data.isna().all():
            min_date = col_data.min()
            max_date = col_data.max()
            info_parts.append(f": Range from {min_date} to {max_date}")
    
    # Add missing value info if any exist
    if missing_count > 0:
        info_parts.append(f", {missing_count} missing values")
    
    return "".join(info_parts)


def _build_missing_values_info(df):
    """
    Build summary information about missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
        
    Returns:
        str: Formatted string with missing value statistics
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    
    if total_missing == 0:
        return "Missing Values: None detected"
    else:
        missing_percentage = (total_missing / total_cells) * 100
        return f"Missing Values: {total_missing} ({missing_percentage:.1f}% of total data)"


def build_prompt_with_sample(df, sample_rows=5):
    """
    Build a prompt that includes sample data rows in addition to statistics.
    
    This is useful when you want to give the AI more context about the actual
    data values, not just the statistics.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
        sample_rows (int): Number of sample rows to include (default: 5)
        
    Returns:
        str: Enhanced prompt with sample data included
    """
    # Get the base prompt
    base_prompt = build_prompt(df)
    
    # Add sample data section
    sample_section = [
        "",
        f"Sample Data (first {min(sample_rows, len(df))} rows):",
        ""
    ]
    
    # Convert sample rows to a readable format
    sample_df = df.head(sample_rows)
    sample_str = sample_df.to_string(index=False)
    sample_section.append(sample_str)
    
    # Combine base prompt with sample data
    enhanced_prompt = base_prompt + "\n" + "\n".join(sample_section)
    
    return enhanced_prompt


# Example usage and testing for whne you wan to run this file directly
if __name__ == "__main__":
    """
    Test the prompt_builder module with sample data.
    This runs only when the script is executed directly.
    """
    print("Testing prompt_builder.py module...\n")
    
    # Create a sample DataFrame for testing
    sample_data = {
        'Age': [28, 35, 42, 31, 45, 29, 38],
        'Salary': [45000, 78000, 65000, 52000, 95000, 48000, 70000],
        'Department': ['Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering', 'HR', 'Marketing'],
        'Experience': [2, 7, 12, 4, 15, 3, 10],
        'Bonus': [2000, 6500, 4500, 3200, 9000, 2800, 5500],
        'Performance_Score': [85, 92, 88, 79, 95, 82, 90]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some missing values for testing
    df.loc[1, 'Bonus'] = np.nan
    df.loc[3, 'Performance_Score'] = np.nan
    
    print("=" * 80)
    print("SAMPLE DATAFRAME:")
    print("=" * 80)
    print(df)
    print("\n")
    
    print("=" * 80)
    print("GENERATED PROMPT:")
    print("=" * 80)
    prompt = build_prompt(df)
    print(prompt)
    print("\n")
    
    print("=" * 80)
    print("PROMPT WITH SAMPLE DATA:")
    print("=" * 80)
    enhanced_prompt = build_prompt_with_sample(df, sample_rows=3)
    print(enhanced_prompt)
    print("\n")
    
    print(f" Prompt generated successfully!")
    print(f" Prompt length: {len(prompt)} characters")
    print(f" Enhanced prompt length: {len(enhanced_prompt)} characters")