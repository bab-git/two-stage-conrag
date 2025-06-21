from typing import Union
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Import Groq integration
try:
    from langchain_groq import ChatGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

import logging
import os
import streamlit as st

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Enhanced LLM Manager that supports both OpenAI and local models (via llama-cpp).

    The class can automatically detect the model type and use the appropriate inference method.
    It supports both string prompts and LangChain PromptTemplate objects.
    """

    def __init__(self, model_config: dict, api_key: str = None):
        """
        Initialize the LLMManager with a specific LLM instance.

        Args:
            llm_instance: Either a ChatOpenAI instance or Llama instance
            model_type: "auto", "openai", or "llama_cpp" - determines inference method
        """
        self.model_config = model_config
        self.provider = model_config.get("provider")
        self.model_id = model_config.get("model_id")
        self.requires_key = model_config.get("requires_key", False)

        # Initialize the appropriate LLM
        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model_id, temperature=0, api_key=api_key)
            self.model_type = "openai"

        elif self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError(
                    "langchain-groq is not installed. Install it with: pip install langchain-groq"
                )

            # Get Groq API key from secrets (for cloud) or environment (for local)
            groq_key = self._get_groq_api_key()
            if not groq_key:
                raise ValueError(
                    "Groq API key not found. Please check your environment variables or Streamlit secrets."
                )

            self.llm = ChatGroq(model=self.model_id, temperature=0, api_key=groq_key)
            self.model_type = "groq"

        elif self.provider == "llama_cpp":
            # For local LLaMA models, llm_instance should be passed separately
            self.llm = None  # Will be set later
            self.model_type = "llama_cpp"

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_groq_api_key(self) -> str:
        """Get Groq API key from Streamlit secrets or environment variables."""
        # Try Streamlit secrets first (for cloud deployment)
        try:
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                return st.secrets["GROQ_API_KEY"]
        except (AttributeError, KeyError):
            pass

        # Fall back to environment variable
        return os.getenv("GROQ_API_KEY", "")

    def set_llama_instance(self, llama_instance):
        """Set the LLaMA instance for local models."""
        if self.provider == "llama_cpp":
            self.llm = llama_instance
        else:
            raise ValueError(
                "set_llama_instance can only be called for llama_cpp provider"
            )

    def invoke(
        self,
        prompt: Union[str, PromptTemplate, ChatPromptTemplate],
        invoke_kwargs: dict = None,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Invoke the LLM with the given prompt and parameters.

        Args:
            prompt: Either a string prompt or LangChain PromptTemplate
            llm_instance: Optional LLM instance to use (overrides self.llm)
            **kwargs: Additional parameters for prompt formatting

        Returns:
            str: The generated response
        """
        # Use provided llm_instance or fall back to self.llm
        # current_llm = llm_instance if llm_instance is not None else self.llm

        # Detect model type for the current LLM if needed
        # if llm_instance is not None:
        #     current_model_type = self._detect_model_type(llm_instance)
        # else:
        #     current_model_type = self.model_type

        try:
            if self.model_type in ["openai", "groq"]:
                return self._invoke_langchain(prompt, invoke_kwargs, verbose=verbose)
            elif self.model_type == "llama_cpp":
                return self._invoke_llama_cpp(
                    prompt=prompt,
                    invoke_kwargs=invoke_kwargs,
                    verbose=verbose,
                    max_tokens=generation_kwargs.get("max_tokens", 10000),
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            raise

    def _invoke_langchain(
        self,
        system_prompt: Union[str, PromptTemplate, ChatPromptTemplate],
        invoke_kwargs: dict,
        verbose: bool = False,
    ) -> str:
        """
        Invoke OpenAI or Groq model via LangChain.

        Args:
            system_prompt: Prompt template or string
            invoke_kwargs: Parameters for prompt formatting
            verbose: Enable verbose logging

        Returns:
            str: Generated response
        """
        # if isinstance(prompt, str):
        #     # Convert string to PromptTemplate
        #     prompt_template = PromptTemplate.from_template(prompt)
        # else:
        #     prompt_template = prompt
        prompt_template = PromptTemplate.from_template(system_prompt)
        # chain = prompt_template | self.llm | StrOutputParser()
        # Create the chain
        chain = prompt_template | self.llm | StrOutputParser()

        # Invoke the chain
        response = chain.invoke(invoke_kwargs)
        response = response.strip()

        if verbose:
            print(f"LLM Response from {self.provider}: {response[:100]}...")

        return response

    def _invoke_llama_cpp(
        self,
        prompt: Union[str, PromptTemplate],
        invoke_kwargs: dict,
        verbose: bool,
        max_tokens: int,
        stop_tokens: list = None,
    ) -> str:
        """
        Invoke llama-cpp model.

        Args:
            llm_instance: Llama instance
            prompt: Prompt template or string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Stop sequences
            **kwargs: Parameters for prompt formatting

        Returns:
            str: Generated response
        """

        # Format the prompt if it's a template
        if invoke_kwargs is not None:
            # print("is prompt template")
            formatted_prompt = prompt.format(**invoke_kwargs)
        else:
            formatted_prompt = prompt

        # Set default stop sequences if none provided
        if stop_tokens is None:
            stop_tokens = ["\n\n", "</s>", "<|im_end|>"]

        if verbose:
            print("====== DEBUG: LLM Instance ID:", id(self.llm))
            print("====== DEBUG: Model Path:", getattr(self.llm, "model_path", "N/A"))
            print("====== DEBUG: Context Size:", self.llm.n_ctx())
            print("====== DEBUG: Formatted Prompt:")
            print(repr(formatted_prompt))  # Use repr to show exact characters
            print("====== DEBUG: Stop Tokens:", stop_tokens)
            print("====== DEBUG: Max Tokens:", max_tokens)

        # RESET MODEL CONTEXT TO ENSURE CLEAN STATE
        # This is important for cached models in Streamlit
        if hasattr(self.llm, "reset"):
            self.llm.reset()
            if verbose:
                print("====== DEBUG: Model instance reset")
        else:
            if verbose:
                print("====== DEBUG: Model instance does not support reset")

        # Generate response
        response = self.llm.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=0,
            stop=stop_tokens,
        )

        raw_response = response["choices"][0]["text"]
        stripped_response = raw_response.strip()

        # ADD MORE DEBUGGING
        if verbose:
            print("====== DEBUG: Raw Response:")
            print(repr(raw_response))  # Show exact response with whitespace
            print("====== DEBUG: Stripped Response:")
            print(repr(stripped_response))
            print("====== DEBUG: Response Length:", len(stripped_response))
            print(
                "====== DEBUG: Finish Reason:",
                response["choices"][0].get("finish_reason", "N/A"),
            )

        return stripped_response
