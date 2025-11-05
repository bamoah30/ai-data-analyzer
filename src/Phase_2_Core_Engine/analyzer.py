"""
analyzer.py - Send prompts to AI API (Hugging Face or OpenAI) and retrieve insights

This module handles communication with AI APIs to generate data analysis insights from
structured prompts. It supports both Hugging Face and OpenAI APIs, with a proven working
implementation using the Hugging Face router endpoint.

Key features:
    - Sends structured prompts to AI APIs for analysis
    - Returns formatted AI-generated insights
    - Supports both Hugging Face (free) and OpenAI (paid) providers
    - Robust error handling with informative error messages
    - Loads API keys from .env file or environment variables
    - Typical response time: 2-5 seconds
    - Multiple response format parsing for compatibility
"""

import os
import json
import requests
from typing import Optional, List, Dict
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Hugging Face API endpoint - uses router for model selection
HUGGINGFACE_API_URL = "https://router.huggingface.co/v1/chat/completions"


def _extract_text_from_choice(choice: Dict) -> Optional[str]:
    """
    Extract text content from various response formats returned by Hugging Face API.
    
    The Hugging Face API can return responses in multiple formats depending on the model.
    This function handles several common shapes to ensure compatibility:
    - Chat completion format (message with content)
    - Older completion format (direct text field)
    - Alternative formats (parts, segments, output)
    
    Args:
        choice (dict): A single choice object from the API response
        
    Returns:
        Optional[str]: The extracted text content, or None if extraction fails
    """
    if not isinstance(choice, dict):
        return None

    # Chat completion format: message can be string or dict
    msg = choice.get("message")
    if isinstance(msg, str):
        return msg
    
    if isinstance(msg, dict):
        # Try standard content field
        content = msg.get("content")
        if isinstance(content, str):
            return content
        
        # Handle nested content structures (parts, text, segments)
        if isinstance(content, dict):
            parts = content.get("parts") or content.get("text") or content.get("segments")
            if isinstance(parts, list):
                return "".join(str(p) for p in parts)
            if isinstance(parts, str):
                return parts
            
            # Try direct text key
            text = content.get("text")
            if isinstance(text, str):
                return text

    # Older completion format with top-level text field
    text = choice.get("text")
    if isinstance(text, str):
        return text

    return None


def _query_huggingface(payload: Dict, api_key: str, timeout: int = 30) -> str:
    """
    Send a request to Hugging Face API and parse the response with format tolerance.
    
    This function handles the low-level communication with Hugging Face API and
    gracefully parses various response formats. It's designed to be robust against
    different model output formats.
    
    Args:
        payload (dict): The request payload containing messages and model parameters
        api_key (str): Hugging Face API token for authentication
        timeout (int): Request timeout in seconds (default: 30)
        
    Returns:
        str: Extracted text response from the API
        
    Raises:
        RuntimeError: If the API request fails or returns invalid data
    """
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Hugging Face API request timed out. The model may be loading. "
            "Please try again in a few moments."
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Failed to connect to Hugging Face API. Check your internet connection: {str(e)}"
        )
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise RuntimeError(
                "Invalid Hugging Face API token. Verify your token at "
                "https://huggingface.co/settings/tokens"
            )
        elif "403" in error_msg:
            raise RuntimeError("Access denied to the specified model.")
        elif "429" in error_msg or "Too Many Requests" in error_msg:
            raise RuntimeError(
                "Hugging Face API rate limit exceeded. Please wait and try again."
            )
        else:
            raise RuntimeError(f"Hugging Face API error: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error communicating with Hugging Face API: {str(e)}")

    # Parse response with tolerance for different formats
    try:
        data = response.json()
    except ValueError:
        # If response is not JSON, return raw text
        return response.text

    # Try chat completion format: choices -> message -> content
    if isinstance(data, dict):
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            text = _extract_text_from_choice(choices[0])
            if text:
                return text

        # Try alternative top-level keys
        for key in ("generated_text", "output", "response", "result"):
            val = data.get(key)
            if isinstance(val, str):
                return val

        # Check for error response
        error = data.get("error", {})
        if error:
            error_msg = error.get("message", str(error))
            raise RuntimeError(f"API returned error: {error_msg}")

    # Fallback: return JSON string representation
    return json.dumps(data)


def get_insights(
    prompt: str,
    api_key: Optional[str] = None,
    provider: str = "huggingface",
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """
    Send a prompt to an AI API and retrieve AI-generated insights.
    
    Main entry point for generating insights. Supports multiple providers and models,
    with comprehensive validation and error handling. API keys can be passed directly
    or loaded from environment variables.
    
    Args:
        prompt (str): The structured prompt to send to the API
        api_key (Optional[str]): API key for the provider. If None, reads from environment.
                                For Hugging Face: HUGGINGFACE_API_KEY
                                For OpenAI: OPENAI_API_KEY
        provider (str): AI provider to use - "huggingface" (default) or "openai"
        model (Optional[str]): Model to use. If None, uses provider default:
                              - Hugging Face: MiniMaxAI/MiniMax-M2:novita
                              - OpenAI: "gpt-3.5-turbo"
        max_tokens (int): Maximum tokens in the response (default: 1000)
        temperature (float): Response creativity 0.0-1.0 (default: 0.7)
                            - Lower values (0.0-0.3): More focused and deterministic
                            - Higher values (0.7-1.0): More creative and diverse
        
    Returns:
        str: AI-generated insights as a formatted string
        
    Raises:
        ValueError: If API key is missing, prompt is empty, or provider is invalid
        RuntimeError: If API call fails (rate limits, network issues, etc.)
    """
    # Validate prompt input
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise ValueError("Prompt must be a non-empty string")

    # Validate and normalize provider
    provider = provider.lower()
    if provider not in ["huggingface", "openai"]:
        raise ValueError("Provider must be either 'huggingface' or 'openai'")

    # Determine which environment variable to use
    env_var = "HUGGINGFACE_API_KEY" if provider == "huggingface" else "OPENAI_API_KEY"

    # Get API key: use provided key or fall back to environment variable
    if api_key is None:
        api_key = os.getenv(env_var)

    # Ensure API key is available
    if not api_key:
        provider_name = "Hugging Face" if provider == "huggingface" else "OpenAI"
        raise ValueError(
            f"{provider_name} API key is required. Provide it as an argument or set "
            f"the {env_var} environment variable. You can also create a .env file "
            f"in your project root with: {env_var}=your-key-here"
        )

    # Route to appropriate provider
    if provider == "huggingface":
        return _get_insights_huggingface(prompt, api_key, model, max_tokens, temperature)
    else:
        return _get_insights_openai(prompt, api_key, model, max_tokens, temperature)


def _get_insights_huggingface(
    prompt: str,
    api_key: str,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """
    Send a prompt to Hugging Face API and retrieve AI-generated insights.
    
    Uses the Hugging Face router endpoint which automatically selects an appropriate
    model. This is the recommended implementation as it's free and works reliably
    with various models.
    
    Args:
        prompt (str): The structured prompt for analysis
        api_key (str): Hugging Face API token
        model (Optional[str]): Model to use (default is MiniMaxAI/MiniMax-M2:novita)
        max_tokens (int): Maximum tokens in response (default: 1000)
        temperature (float): Response creativity level (default: 0.7)
        
    Returns:
        str: AI-generated insights
        
    Raises:
        RuntimeError: If API call fails
    """
    # Build the system prompt for data analysis context
    system_message = (
        "You are a data analyst expert. Analyze the provided dataset summary "
        "and provide clear, actionable insights about trends, patterns, correlations, "
        "and data quality issues. Structure your response with clear sections and "
        "bullet points where appropriate."
    )

    # Construct the API request payload
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
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Add model if specified, otherwise let router decide
    if model:
        # Extract model name if full path provided
        payload["model"] = model.split("/")[-1] if "/" in model else model
    else:
        # Router will select an appropriate model automatically
        payload["model"] = "MiniMaxAI/MiniMax-M2:novita"

    # Send request and handle response
    try:
        insights = _query_huggingface(payload, api_key)
        return insights.strip() if insights else ""
    except RuntimeError as e:
        raise RuntimeError(f"Hugging Face API error: {str(e)}")


def _get_insights_openai(
    prompt: str,
    api_key: str,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """
    Send a prompt to OpenAI API and retrieve AI-generated insights.
    
    Uses the OpenAI client library for reliable communication with OpenAI endpoints.
    Requires a valid paid OpenAI API key.
    
    Args:
        prompt (str): The structured prompt for analysis
        api_key (str): OpenAI API key
        model (Optional[str]): Model to use (default: "gpt-3.5-turbo")
        max_tokens (int): Maximum tokens in response (default: 1000)
        temperature (float): Response creativity level (default: 0.7)
        
    Returns:
        str: AI-generated insights
        
    Raises:
        RuntimeError: If API call fails
    """
    # Use default model if not specified
    if model is None:
        model = "gpt-3.5-turbo"

    # Build the system prompt for data analysis context
    system_message = (
        "You are a data analyst expert. Analyze the provided dataset summary "
        "and provide clear, actionable insights about trends, patterns, correlations, "
        "and data quality issues. Structure your response with clear sections and "
        "bullet points where appropriate."
    )

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Send request to OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Extract response content
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()

        raise RuntimeError("No content received from OpenAI API. Response was empty.")

    except Exception as e:
        error_msg = str(e)

        # Provide user-friendly error messages for common issues
        if "rate_limit" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API rate limit exceeded. Please wait a few minutes before trying again. "
                f"Details: {error_msg}"
            )
        elif "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower():
            raise RuntimeError(
                "Invalid OpenAI API key. Check your key at https://platform.openai.com/api-keys"
            )
        elif "insufficient_quota" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API quota exceeded. Check your billing at "
                "https://platform.openai.com/account/billing"
            )
        else:
            raise RuntimeError(f"OpenAI API error: {error_msg}")


def get_insights_with_history(
    prompt: str,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
    provider: str = "huggingface",
    model: Optional[str] = None
) -> tuple:
    """
    Send a prompt with conversation history for follow-up questions.
    
    Maintains conversation context across multiple API calls. Note that Hugging Face
    constructs history locally, while OpenAI maintains context server-side.
    
    Args:
        prompt (str): The current prompt or question
        api_key (Optional[str]): API key for the provider
        conversation_history (Optional[List[Dict]]): Previous message dicts in format:
                                                    [{"role": "user", "content": "..."}, ...]
        provider (str): AI provider - "huggingface" or "openai"
        model (Optional[str]): Model to use
        
    Returns:
        tuple: (insights_string, updated_conversation_history)
        
    Raises:
        ValueError: If API key is missing or provider is invalid
        RuntimeError: If API call fails
    """
    # Determine environment variable for API key
    env_var = "HUGGINGFACE_API_KEY" if provider == "huggingface" else "OPENAI_API_KEY"

    # Get API key from argument or environment
    if api_key is None:
        api_key = os.getenv(env_var)

    if not api_key:
        raise ValueError(f"API key required. Set {env_var} environment variable.")

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

    # Add current user message to history
    conversation_history.append({"role": "user", "content": prompt})

    try:
        if provider == "openai":
            # OpenAI maintains history server-side
            if model is None:
                model = "gpt-3.5-turbo"

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=conversation_history
            )

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    insights = content.strip()
                    conversation_history.append({"role": "assistant", "content": insights})
                    return insights, conversation_history

            raise RuntimeError("No content received from OpenAI API.")

        else:
            # For Hugging Face, construct conversation context locally
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
        raise RuntimeError(f"Error retrieving insights: {str(e)}")


def estimate_cost(
    prompt: str,
    provider: str = "huggingface",
    model: Optional[str] = None
) -> Dict:
    """
    Estimate the cost of analyzing a prompt.
    
    Provides cost estimation based on token counts and current API pricing.
    Hugging Face is free; OpenAI pricing varies by model.
    
    Note: This is a rough estimate. Actual token counts may vary based on
    encoding differences and model-specific tokenization rules.
    
    Args:
        prompt (str): The prompt to estimate cost for
        provider (str): AI provider - "huggingface" or "openai"
        model (Optional[str]): The model being used (for accurate pricing)
        
    Returns:
        Dict: Cost estimation information including:
              - provider: Selected provider name
              - model: Model being used
              - estimated_input_tokens: Estimated input token count
              - estimated_output_tokens: Estimated output token count
              - estimated_cost_usd: Total estimated cost in USD
              - note: Additional information about the estimate
    """
    # Rough estimate: ~4 characters per token
    estimated_input_tokens = len(prompt) // 4
    estimated_output_tokens = 500

    if provider == "huggingface":
        return {
            "provider": "huggingface",
            "model": model or "Auto-selected via router",
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": 0.0,
            "note": "Hugging Face API is completely free with no payment required."
        }

    elif provider == "openai":
        # Use default model if not specified
        if model is None:
            model = "gpt-3.5-turbo"

        # Current OpenAI pricing (as of knowledge cutoff)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo-preview": {"input": 0.01 / 1000, "output": 0.03 / 1000}
        }

        if model not in pricing:
            return {"error": f"Pricing not available for model: {model}"}

        # Calculate total cost
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
    Test the analyzer module with a sample dataset analysis prompt.
    Run this script directly to test API connectivity and response quality.
    """
    print("Testing analyzer.py module\n")

    # Sample dataset for analysis
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

    # Check for available API keys
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
        # Select provider and API key
        if hf_token:
            print("✓ Hugging Face API token found")
            provider = "huggingface"
            api_key = hf_token
        else:
            print("✓ OpenAI API key found")
            provider = "openai"
            api_key = openai_key

        # Display cost estimate
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
                print(f"\n✗ Error: {e}")
        else:
            print("\nSkipped API call")

    print("\n" + "=" * 80)
    print("MODULE TEST COMPLETE")
    print("=" * 80)