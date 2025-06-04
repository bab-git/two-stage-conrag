import os
from dotenv import load_dotenv, find_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Load secrets from .env
load_dotenv(find_dotenv(), override=True)

def get_env_secrets() -> dict[str, str]:
    """
    Get environment secrets from .env file.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if openai_api_key and openai_api_key.startswith('sk-proj-') and len(openai_api_key)>10:
        print("OpenAI API key looks good so far, ends with:", openai_api_key[-10:])        
    else:
        raise RuntimeError("Please set a proper OPENAI_API_KEY in your .env file or environment.")
    
    # check if langsmith_api_key is loaded
    if langsmith_api_key:
        print("LangSmith API key found, ends with:", langsmith_api_key[-10:])
    else:
        print("LangSmith API key not found")

    return {
        "OPENAI_API_KEY": openai_api_key,
        "LANGSMITH_API_KEY": langsmith_api_key,
    }    


# Find if streamlit is running
def is_streamlit_running() -> bool:
    """
    Checks if the script is running within a Streamlit app.

    Returns:
        bool: True if running in Streamlit, False otherwise.
    """    
    try:        
        return get_script_run_ctx() is not None
    except:
        return False

streamlit_running = is_streamlit_running()
if streamlit_running == False:
    print('streamlit is not running')
else:
    print('streamlit is running')
