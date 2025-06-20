from typing import Union
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# try:
#     from llama_cpp import Llama
#     LLAMA_CPP_AVAILABLE = True
# except ImportError:
#     LLAMA_CPP_AVAILABLE = False
#     Llama = None

import logging

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Enhanced LLM Manager that supports both OpenAI and local models (via llama-cpp).

    The class can automatically detect the model type and use the appropriate inference method.
    It supports both string prompts and LangChain PromptTemplate objects.
    """

    def __init__(self, llm_instance=None):
        """
        Initialize the LLMManager with a specific LLM instance.

        Args:
            llm_instance: Either a ChatOpenAI instance or Llama instance
            model_type: "auto", "openai", or "llama_cpp" - determines inference method
        """
        if llm_instance is None:
            # Default to OpenAI if no instance provided
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
            self.model_type = "openai"
        else:
            self.llm = llm_instance
            self.model_type = "llama_cpp"

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
            if self.model_type == "openai":
                return self._invoke_openai(prompt, invoke_kwargs)
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

    def _invoke_openai(
        self,
        system_prompt: Union[str, PromptTemplate, ChatPromptTemplate],
        invoke_kwargs: dict,
    ) -> str:
        """
        Invoke OpenAI model via LangChain.

        Args:
            llm_instance: ChatOpenAI instance
            prompt: Prompt template or string
            **kwargs: Parameters for prompt formatting

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
