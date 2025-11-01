"""
analyzer.py - Send prompts to an AI API (Hugging Face or OpenAI) and retrieve AI-generated insights

This module handles communication with AI APIs to generate data analysis
insights from structured prompts. It supports both Hugging Face and OpenAI
APIs, allowing users to choose their preferred provider.

Key features:
- Sends structured prompts to AI APIs
- Returns formatted AI-generated insights
- Supports both Hugging Face (default, free) and OpenAI (paid)
- Configurable model selection
- Robust error handling for API issues
- Loads API key from .env file or environment variables
- Typical response time: 2-5 seconds
"""

import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional, List


# Load environment variables from .env file
load_dotenv()

# Hugging Face API endpoint
# HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# HUGGINGFACE_API_URL = # Replace this:
# HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# With this:
# HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
#HUGGINGFACE_API_URL = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_API_URL ="https://mistralai/Mistral-7B-Instruct-v0.1"


def get_insights(prompt:str, 
                 api_key: Optional[str] = None, 
                 provider:str="huggingface", 
                 model: Optional[str] = None, 
                 max_tokens:int=1000, 
                 temperature:float=0.7) ->str:
    """
    Send a prompt to an AI API and retrieve AI-generated insights.
    
    Args:
        prompt (str): The structured prompt to send to the API
        api_key (str, optional): API key for the chosen provider. 
                                If None, reads from environment variable
        provider (str, optional): AI provider to use - "huggingface" (default) or "openai"
        model (str, optional): Model to use. If None, uses provider default:
                              - Hugging Face: mistralai/Mistral-7B-Instruct-v0.1
                              - OpenAI: gpt-3.5-turbo
        max_tokens (int, optional): Maximum tokens in the response (default: 1000)
        temperature (float, optional): Response creativity 0.0-1.0 (default: 0.7)
                                      Lower = more focused, Higher = more creative
        
    Returns:
        str: AI-generated insights as a formatted string
        
    Raises:
        ValueError: If API key is missing, prompt is empty, or provider is invalid
        Exception: If API call fails (rate limits, network issues, etc.)
    """
    # Validate inputs
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("Prompt must be a non-empty string")
    
    provider = provider.lower()
    if provider not in ["huggingface", "openai"]:
        raise ValueError("Provider must be either 'huggingface' or 'openai'")
    
    env_var = "HUGGINGFACE_API_KEY" if provider == "huggingface" else "OPENAI_API_KEY"


    # Get API key based on provider
    if api_key is None:
        api_key = os.getenv(env_var)
    
    if not api_key:
        provider_name = "Hugging Face" if provider == "huggingface" else "OpenAI"
        raise ValueError(
            f"{provider_name} API key is required. Either pass it as an argument or set the "
            f"{env_var} environment variable. You can also create a .env file "
            f"in your project root with: {env_var}=your-key-here"
        )
    
    if provider == "huggingface":
        return _get_insights_huggingface(prompt, api_key, model, max_tokens, temperature)
    else:
        return _get_insights_openai(prompt, api_key, model, max_tokens, temperature)


def _get_insights_openai(prompt, api_key, model=None, max_tokens=1000, temperature=0.7):
    """
    Send a prompt to OpenAI API and retrieve AI-generated insights.
    
    Args:
        prompt (str): The structured prompt
        api_key (str): OpenAI API key
        model (str, optional): Model to use (default: "gpt-3.5-turbo")
        max_tokens (int, optional): Maximum tokens in response
        temperature (float, optional): Response creativity level
        
    Returns:
        str: AI-generated insights
        
    Raises:
        Exception: If API call fails
    """
    if model is None:
        model = "gpt-3.5-turbo"
    
    try:
        client = OpenAI(api_key=api_key)
        
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
        
        if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
            insights = response.choices[0].message.content.strip()
            return insights
        else:
            raise Exception("No content received from OpenAI API. The response was empty or malformed.")
    
    except Exception as e:
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


def _get_insights_huggingface(prompt, api_key, model=None, max_tokens=1000, temperature=0.7):
    """
    Send a prompt to Hugging Face API and retrieve AI-generated insights.
    
    Args:
        prompt (str): The structured prompt
        api_key (str): Hugging Face API token
        model (str, optional): Model to use (default: mistralai/Mistral-7B-Instruct-v0.1)
        max_tokens (int, optional): Maximum tokens in response
        temperature (float, optional): Response creativity level
        
    Returns:
        str: AI-generated insights
        
    Raises:
        Exception: If API call fails
    """
    if model is None:
        model =  HUGGINGFACE_API_URL #model = "mistralai/Mistral-7B-Instruct-v0.
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    system_message = (
        "You are a data analyst expert. Analyze the provided dataset summary "
        "and provide clear, actionable insights about trends, patterns, correlations, "
        "and data quality issues. Structure your response with clear sections and "
        "bullet points where appropriate."
    )
    
    full_prompt = f"{system_message}\n\nDataset Analysis Request:\n{prompt}"
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model.split("/")[-1] if "/" in model else model,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(model, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Handle chat completion response format
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return content.strip()
                elif isinstance(message, str):
                    return message.strip()
            
            # Check for error messages
            error = result.get("error", {})
            if error:
                error_msg = error.get("message", str(error))
                raise Exception(error_msg)
        
        raise Exception(f"Unexpected response format from Hugging Face API: {result}")
    
    except requests.exceptions.Timeout:
        raise Exception(
            "Hugging Face API request timed out. The model may be loading. "
            "Please try again in a few moments."
        )
    except requests.exceptions.ConnectionError as e:
        raise Exception(
            f"Failed to connect to Hugging Face API. Please check your internet connection. "
            f"Original error: {str(e)}"
        )
    except requests.exceptions.HTTPError as e:
        error_message = str(e)
        if "401" in error_message or "Unauthorized" in error_message:
            raise Exception(
                "Invalid Hugging Face API token. Please check your token at "
                "https://huggingface.co/settings/tokens"
            )
        elif "403" in error_message:
            raise Exception(
                "Access denied to the specified model. Please ensure you have access to the model."
            )
        elif "429" in error_message or "Too Many Requests" in error_message:
            raise Exception(
                "Hugging Face API rate limit exceeded. Please wait a few moments and try again."
            )
        else:
            raise Exception(f"Hugging Face API error: {error_message}")
    except Exception as e:
        raise Exception(f"Hugging Face API error: {str(e)}")


def get_insights_with_history(prompt: str, api_key: Optional[str] = None, 
                             conversation_history: Optional[List] = None, 
                             provider: str = "huggingface", 
                             model: Optional[str] = None):
    """
    Send a prompt with conversation history for follow-up questions.
    
    Note: This feature is primarily designed for OpenAI. Hugging Face support
    is limited as the API does not maintain conversation history server-side.
    
    Args:
        prompt (str): The current prompt or question
        api_key (str, optional): API key for the chosen provider
        conversation_history (list, optional): List of previous message dicts
        provider (str, optional): AI provider - "huggingface" or "openai"
        model (str, optional): Model to use
        
    Returns:
        tuple: (insights_string, updated_conversation_history)
        
    Raises:
        ValueError: If API key is missing or provider is invalid
        Exception: If API call fails
    """

    env_var = "HUGGINGFACE_API_KEY" if provider == "huggingface" else "OPENAI_API_KEY"

    if api_key is None:
        api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(f"API key is required. Set {env_var} environment variable.")
    
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
    
    conversation_history.append({"role": "user", "content": prompt})
    
    try:
        if provider == "openai":
            if model is None:
                model = "gpt-3.5-turbo"
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history
            )
            
            if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
                insights = response.choices[0].message.content.strip()
                conversation_history.append({"role": "assistant", "content": insights})
                return insights, conversation_history
            else:
                raise Exception("No content received from OpenAI API.")
        
        else:
            # For Hugging Face, construct full conversation context
            messages_text = ""
            for msg in conversation_history:
                if msg["role"] != "system":
                    messages_text += f"\n{msg['role'].upper()}: {msg['content']}"
            
            insights = get_insights(
                messages_text,
                api_key=api_key,
                provider="huggingface",
                model=model
            )
            
            conversation_history.append({"role": "assistant", "content": insights})
            return insights, conversation_history
    
    except Exception as e:
        raise Exception(f"Error retrieving insights: {str(e)}")


def estimate_cost(prompt: str, provider: str = "huggingface", model: Optional[str] = None):
    """
    Estimate the cost of analyzing a prompt.
    
    Note: This is a rough estimate based on typical token usage.
    Actual costs may vary.
    
    Args:
        prompt (str): The prompt to estimate cost for
        provider (str): AI provider - "huggingface" or "openai"
        model (str, optional): The model being used
        
    Returns:
        dict: Cost estimation information
    """
    estimated_input_tokens = len(prompt) // 4
    estimated_output_tokens = 500
    
    if provider == "huggingface":
        return {
            "provider": "huggingface",
            "model": model or "mistralai/Mistral-7B-Instruct-v0.1",
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": 0.0,
            "note": "Hugging Face API is completely free with no payment required."
        }
    
    elif provider == "openai":
        if model is None:
            model = "gpt-3.5-turbo"
        
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
            "provider": "openai",
            "model": model,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": round(total_cost, 4),
            "note": "This is a rough estimate. Actual costs may vary."
        }
    
    else:
        return {"error": f"Unknown provider: {provider}"}


if __name__ == "__main__":
    """
    Test the analyzer module with a sample prompt.
    This runs only when the script is executed directly.
    """
    print("Testing analyzer.py module\n")
    
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
    
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not hf_token and not openai_key:
        print("No API keys found in environment variables")
        print("\nTo test this module, set one of these options:")
        print("\n1. Hugging Face (Recommended - Free):")
        print("   export HUGGINGFACE_API_KEY=your-token-here")
        print("   Or create a .env file with: HUGGINGFACE_API_KEY=your-token-here")
        print("\n2. OpenAI (Paid):")
        print("   export OPENAI_API_KEY=your-key-here")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
    else:
        if hf_token:
            print("Hugging Face API token found")
            provider = "huggingface"
            api_key = hf_token
        else:
            print("OpenAI API key found")
            provider = "openai"
            api_key = openai_key
        
        cost_info = estimate_cost(sample_prompt, provider=provider)
        print(f"\nProvider: {cost_info['provider']}")
        print(f"Model: {cost_info['model']}")
        print(f"Estimated cost: ${cost_info['estimated_cost_usd']}")
        print(f"Input tokens: ~{cost_info['estimated_input_tokens']}")
        print(f"Output tokens: ~{cost_info['estimated_output_tokens']}")
        
        print("\n" + "=" * 80)
        user_input = input("Would you like to send this prompt to the API? (y/n): ")
        
        if user_input.lower() == 'y':
            try:
                print("\nSending request to API...")
                print("This may take 2-5 seconds\n")
                
                insights = get_insights(
                    sample_prompt,
                    api_key=api_key,
                    provider=provider
                )
                
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