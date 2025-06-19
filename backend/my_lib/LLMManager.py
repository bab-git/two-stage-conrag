from typing import Union, Dict, Any
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
            self.model_type = 'llama_cpp'        
    
    def invoke(self, prompt, invoke_kwargs) -> str:
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
                return self._invoke_llama_cpp(prompt, invoke_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            raise
    
    def _invoke_openai(self, system_prompt: Union[str, PromptTemplate, ChatPromptTemplate], 
                      invoke_kwargs: dict) -> str:
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
    
    def _invoke_llama_cpp(self, llm_instance, prompt: Union[str, PromptTemplate], 
                         max_tokens: int = 512, temperature: float = 0.0, 
                         stop: list = None, **kwargs) -> str:
        """
        Invoke llama-cpp model.
        
        Args:
            llm_instance: Llama instance
            prompt: Prompt template or string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Parameters for prompt formatting
            
        Returns:
            str: Generated response
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Please install it to use local models.")
        
        # Format the prompt if it's a template
        if isinstance(prompt, PromptTemplate):
            formatted_prompt = prompt.format(**kwargs)
        elif isinstance(prompt, str) and kwargs:
            # Handle string prompts with format parameters
            formatted_prompt = prompt.format(**kwargs)
        else:
            formatted_prompt = prompt
        
        # Set default stop sequences if none provided
        if stop is None:
            stop = ["\n\n", "</s>", "<|im_end|>"]
        
        # Generate response
        response = llm_instance.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        
        return response["choices"][0]["text"].strip()
    
    def set_llm(self, llm_instance, model_type: str = "auto"):
        """
        Set a new LLM instance.
        
        Args:
            llm_instance: New LLM instance
            model_type: "auto", "openai", or "llama_cpp"
        """
        self.llm = llm_instance
        self.model_type = self._detect_model_type(llm_instance) if model_type == "auto" else model_type
    
    @classmethod
    def create_openai_manager(cls, model: str = "gpt-4o", temperature: float = 0.0):
        """
        Factory method to create an LLMManager with OpenAI.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            
        Returns:
            LLMManager: Configured manager instance
        """
        llm = ChatOpenAI(model=model, temperature=temperature)
        return cls(llm, "openai")
    
    @classmethod
    def create_llama_cpp_manager(cls, model_path: str = None, repo_id: str = None, 
                                filename: str = None, **llama_kwargs):
        """
        Factory method to create an LLMManager with llama-cpp.
        
        Args:
            model_path: Path to local model file
            repo_id: Hugging Face repo ID
            filename: Model filename pattern
            **llama_kwargs: Additional arguments for Llama initialization
            
        Returns:
            LLMManager: Configured manager instance
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Please install it to use local models.")
        
        if model_path:
            llm = Llama(model_path=model_path, **llama_kwargs)
        elif repo_id and filename:
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **llama_kwargs
            )
        else:
            raise ValueError("Either model_path or (repo_id + filename) must be provided")
        
        return cls(llm, "llama_cpp")