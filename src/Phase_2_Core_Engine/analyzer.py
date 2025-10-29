"""
analyzer.py - Send prompts to OpenAI API and retrieve AI-generated insights

This module handles communication with the OpenAI API to generate data analysis
insights from structured prompts. It uses GPT models to provide intelligent
interpretations of dataset summaries.

Key features:
- Sends structured prompts to OpenAI API
- Returns formatted AI-generated insights
- Configurable model selection (default: gpt-3.5-turbo)
- Robust error handling for API issues
- Loads API key from .env file or environment variables
- Typical response time: 2-5 seconds
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional, List


# Load environment variables from .env file
load_dotenv()


def get_insights(prompt, api_key=None, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7):
    """
    Send a prompt to OpenAI API and retrieve AI-generated insights.
    
    Args:
        prompt (str): The structured prompt to send to OpenAI
        api_key (str, optional): OpenAI API key. If None, reads from OPENAI_API_KEY env variable or .env file
        model (str, optional): OpenAI model to use (default: "gpt-3.5-turbo")
                              Other options: "gpt-4", "gpt-4-turbo-preview"
        max_tokens (int, optional): Maximum tokens in the response (default: 1000)
        temperature (float, optional): Response creativity 0.0-1.0 (default: 0.7)
                                      Lower = more focused, Higher = more creative
        
    Returns:
        str: AI-generated insights as a formatted string
        
    Raises:
        ValueError: If API key is missing or prompt is empty
        Exception: If API call fails (rate limits, network issues, etc.)
    """
    # Validate inputs
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("Prompt must be a non-empty string")
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Either pass it as an argument or set the "
            "OPENAI_API_KEY environment variable. You can also create a .env file "
            "in your project root with: OPENAI_API_KEY=your-key-here"
        )
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create the chat completion request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst expert. Analyze the provided dataset summary "
                        "and provide clear, actionable insights about trends, patterns, correlations, "
                        "and data quality issues. Structure your response with clear sections and "
                        "bullet points where appropriate."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract the insights from the response with safety check
        if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
            insights = response.choices[0].message.content.strip()
            return insights
        else:
            raise Exception("No content received from OpenAI API. The response was empty or malformed.")
    
    except Exception as e:
        # Provide helpful error messages for common issues
        error_message = str(e)
        
        if "rate_limit" in error_message.lower():
            raise Exception(
                "OpenAI API rate limit exceeded. Please wait a few minutes before trying again. "
                f"Original error: {error_message}"
            )
        elif "invalid_api_key" in error_message.lower() or "incorrect api key" in error_message.lower():
            raise Exception(
                "Invalid OpenAI API key. Please check your API key at https://platform.openai.com/api-keys. "
                f"Original error: {error_message}"
            )
        elif "insufficient_quota" in error_message.lower():
            raise Exception(
                "OpenAI API quota exceeded. Please check your billing at https://platform.openai.com/account/billing. "
                f"Original error: {error_message}"
            )
        else:
            raise Exception(f"OpenAI API error: {error_message}")


def get_insights_with_history(prompt: str, api_key: Optional[str]=None, conversation_history:Optional[List]=None, model="gpt-3.5-turbo"):
    """
    Send a prompt with conversation history for follow-up questions.
    
    This function allows for interactive analysis where you can ask follow-up
    questions based on previous insights.
    
    Args:
        prompt (str): The current prompt/question
        api_key (str, optional): OpenAI API key
        conversation_history (list, optional): List of previous message dicts
        model (str, optional): OpenAI model to use
        
    Returns:
        tuple: (insights_str, updated_conversation_history)
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    
    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst expert. Analyze datasets and provide "
                    "clear, actionable insights."
                )
            }
        ]
    
    # Add the new user message
    conversation_history.append({"role": "user", "content": prompt})
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=conversation_history
        )
        
        # Safety check before calling strip()
        if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
            insights = response.choices[0].message.content.strip()
            
            # Add assistant's response to history
            conversation_history.append({"role": "assistant", "content": insights})
            
            return insights, conversation_history
        else:
            raise Exception("No content received from OpenAI API.")
    
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def estimate_cost(prompt:str, model:str="gpt-3.5-turbo"):
    """
    Estimate the cost of analyzing a prompt.
    
    Note: This is a rough estimate based on typical token usage.
    Actual costs may vary.
    
    Args:
        prompt (str): The prompt to estimate cost for
        model (str): The model being used
        
    Returns:
        dict: Estimated cost information
    """
    # Rough token estimation (1 token ≈ 4 characters)
    estimated_input_tokens = len(prompt) // 4
    estimated_output_tokens = 500  # Typical response length
    
    # Pricing as of 2025 (subject to change)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-4-turbo-preview": {"input": 0.01 / 1000, "output": 0.03 / 1000}
    }
    
    if model not in pricing:
        return {"error": f"Pricing not available for model: {model}"}
    
    input_cost = estimated_input_tokens * pricing[model]["input"]
    output_cost = estimated_output_tokens * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "note": "This is a rough estimate. Actual costs may vary."
    }


# Example usage and testing
if __name__ == "__main__":
    """
    Test the analyzer module with a sample prompt.
    This runs only when the script is executed directly.
    """
    print("Testing analyzer.py module...\n")
    
    # Sample prompt (similar to what prompt_builder.py would generate)
    sample_prompt = """Analyze the following dataset summary:

Dataset Shape: 20 rows, 9 columns

Column Information:
- Employee_ID (object): 20 unique values
- Name (object): 20 unique values
- Age (int64): Mean=35.4, Std=7.2, Min=26, Max=50
- Department (object): 4 unique values
- Salary (int64): Mean=63250, Std=16340, Min=43000, Max=95000
- Experience (int64): Mean=7.8, Std=5.6, Min=1, Max=18
- Bonus (int64): Mean=4590, Std=2140, Min=1800, Max=9000
- Performance_Score (int64): Mean=86.2, Std=4.8, Min=78, Max=95
- Hire_Date (object): 20 unique values

Missing Values: None detected

Please provide insights on trends, correlations, and data quality."""
    
    print("=" * 80)
    print("SAMPLE PROMPT:")
    print("=" * 80)
    print(sample_prompt)
    print("\n")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print(" OPENAI_API_KEY not found in environment variables")
        print("\nTo test this module, set your API key in one of these ways:")
        print("\n1. Environment variable:")
        print("   export OPENAI_API_KEY=your-key-here")
        print("\n2. Create a .env file in your project root with:")
        print("   OPENAI_API_KEY=your-key-here")
        print("\n3. Pass the API key directly to get_insights():")
        print("   insights = get_insights(prompt, api_key='your-key-here')")
    else:
        print(" API key found")
        
        # Estimate cost
        cost_info = estimate_cost(sample_prompt)
        print(f"\nEstimated cost: ${cost_info['estimated_cost_usd']}")
        print(f"Input tokens: ~{cost_info['estimated_input_tokens']}")
        print(f"Output tokens: ~{cost_info['estimated_output_tokens']}")
        
        # Ask for confirmation
        print("\n" + "=" * 80)
        user_input = input("Would you like to send this prompt to OpenAI? (y/n): ")
        
        if user_input.lower() == 'y':
            try:
                print("\nSending request to OpenAI...")
                print("This may take 2-5 seconds\n")
                
                insights = get_insights(sample_prompt, api_key=api_key)
                
                print("=" * 80)
                print("AI-GENERATED INSIGHTS:")
                print("=" * 80)
                print(insights)
                print("\n")
                print("Analysis complete!")
                
            except Exception as e:
                print(f"\nError: {e}")
        else:
            print("\nSkipped API call")
    
    print("\n" + "=" * 80)
    print("MODULE TEST COMPLETE")
    print("=" * 80)