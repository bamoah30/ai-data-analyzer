"""
data_loader.py - Load CSV, Excel, and JSON files into pandas DataFrames

This module provides functionality to load various data file formats and convert
them into pandas DataFrames for analysis. It handles common encoding issues and
provides informative error messages for unsupported formats.

Supported formats:
- .csv (Comma-separated values)
- .xlsx (Excel workbooks - single sheet)
- .json (JSON arrays or objects)
"""

import pandas as pd
import os


def load_file(filepath):
    """
    Load a data file and return it as a pandas DataFrame.
    
    Args:
        filepath (str): Path to the data file (.csv, .xlsx, or .json)
        
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame
        
    Raises:
        ValueError: If the file format is unsupported or the file is corrupted
        FileNotFoundError: If the file does not exist
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Get file extension
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()
    
    try:
        # Load CSV files
        if file_extension == '.csv':
            # Try UTF-8 encoding first
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to Latin-1 encoding
                df = pd.read_csv(filepath, encoding='latin-1')
            
            return df
        
        # Load Excel files
        elif file_extension == '.xlsx':
            df = pd.read_excel(filepath)
            return df
        
        # Load JSON files
        elif file_extension == '.json':
            df = pd.read_json(filepath)
            return df
        
        # Unsupported format
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats are: .csv, .xlsx, .json"
            )
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file is empty: {filepath}")
    
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse the file: {filepath}. Error: {str(e)}")
    
    except Exception as e:
        raise ValueError(
            f"An error occurred while loading the file: {filepath}. "
            f"The file may be corrupted or in an invalid format. Error: {str(e)}"
        )
    



# Example usage and testing
if __name__ == "__main__":
    # This section runs only when the script is executed directly
    # It's useful for testing the module during development
    
    print("Testing data_loader.py module...\n")
    
    # Test with a sample CSV file (you'll need to create this)
    #In here please don't forget to use the correct path to your test file within your environment with double backslashes
    test_file = ".\\sample_data\\test.cvs"
    
    if os.path.exists(test_file):
        try:
            df = load_file(test_file)
            print(f" Successfully loaded: {test_file}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nColumn info:")
            print(df.info())
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print(f"Test file not found: {test_file}")
        print("Create a test file to try the module!")

    #Please don't run this whenyou haven't created the test file in the specified path

    
