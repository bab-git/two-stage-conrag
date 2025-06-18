import os
from dotenv import load_dotenv, find_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx
import logging
from pathlib import Path

# Define logs directory and log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)  # create logs/ if it doesn't exist
log_file = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler(log_file, mode="w"),  # single file
    ],
)

logger = logging.getLogger(__name__)

# Load secrets from .env
load_dotenv(find_dotenv(), override=True)


def validate_env_secrets(openai_api_key: str) -> bool:
    """
    Validates that all required environment variables are present and properly formatted.
    Now works with both environment and user-provided API keys.
    """
    # openai_api_key = os.getenv("OPENAI_API_KEY", "")    
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        # raise RuntimeError("OpenAI API key validation failed. This should not happen if the UI logic is working correctly.")
        logger.error("OpenAI API key validation failed. This should not happen if the UI logic is working correctly.")
        return False
    
    logger.info("OpenAI API key validated successfully, ends with %s", openai_api_key[-10:])
    
    if lang_key := os.getenv("LANGCHAIN_API_KEY"):
        logger.info(f"LangSmith key loaded (ends with {lang_key[-10:]})")
    
    return True


# Find if streamlit is running
def is_streamlit_running() -> bool:
    """
    Checks if the script is running within a Streamlit app.

    Returns:
        bool: True if running in Streamlit, False otherwise.
    """
    try:
        return get_script_run_ctx() is not None
    except Exception as e:
        logger.error(f"Error checking if Streamlit is running: {e}")
        return False


# streamlit_running = is_streamlit_running()
if is_streamlit_running():
    logger.info("streamlit is not running")
else:
    logger.info("streamlit is running")
