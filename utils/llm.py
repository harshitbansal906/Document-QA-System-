import os
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    """
    Wrapper for local LLM execution supporting both Seq2Seq (e.g. Flan-T5) 
    and CausalLM (e.g. Qwen, Llama, Phi) models.
    Loads tokenizer and model once on initialization and generates answers offline.
    """
    def __init__(
        self,
        model_id: str = "google/flan-t5-base",
        models_dir: str = None
    ):
        # Resolve path to cache models
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_dir = os.path.join(base_dir, "models", "llm")
        else:
            self.models_dir = os.path.abspath(models_dir)
            
        os.makedirs(self.models_dir, exist_ok=True)

        # Detect GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"LLM computation device: {self.device}")
        
        self.model_id = model_id
        
        # Auto-detect architecture type
        self.is_seq2seq = any(x in model_id.lower() for x in ["t5", "bart", "pegasus"])
        self.model_class = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
        
        logger.info(f"Detected model architecture: {'Seq2Seq (Encoder-Decoder)' if self.is_seq2seq else 'Causal LM (Decoder-only)'}")
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load tokenizer and model from local cache, downloads if missing."""
        try:
            logger.info(f"Attempting to load LLM '{self.model_id}' from local cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.models_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            self.model = self.model_class.from_pretrained(
                self.model_id,
                cache_dir=self.models_dir,
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
            logger.info("Model loaded successfully from local storage.")
        except Exception as e:
            logger.warning(
                f"Could not load '{self.model_id}' offline ({e}). "
                f"Attempting download to '{self.models_dir}'..."
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=self.models_dir,
                    local_files_only=False,
                    trust_remote_code=True
                )
                self.model = self.model_class.from_pretrained(
                    self.model_id,
                    cache_dir=self.models_dir,
                    local_files_only=False,
                    trust_remote_code=True
                ).to(self.device)
                logger.info("Successfully downloaded and initialized model.")
            except Exception as download_err:
                logger.error(f"Failed to download/initialize LLM: {download_err}")
                raise download_err
                
        # Configure pad token for causal models if not set
        if not self.is_seq2seq and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            logger.info("Causal tokenizer pad_token configured to eos_token.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        Generates text completion for a given prompt locally using direct model invocation.
        Returns only the decoded answer.
        
        Args:
            prompt: Formatted prompt string.
            max_new_tokens: Maximum number of tokens to generate. Default 128.
            temperature: Sampling temperature. Default 0.0 (implies greedy/beam search).
            
        Returns:
            The generated response string (only the decoded answer).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM components have not been fully loaded.")

        try:
            # Tokenize input prompt and send to current execution device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Setup generation configuration parameters
            do_sample = temperature > 0.05
            params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_beams": 4  # Beam search with 4 beams
            }
            
            if do_sample:
                params["temperature"] = temperature
                params["top_p"] = 0.95
                
            # If Causal LM, configure padding parameters
            if not self.is_seq2seq:
                params["pad_token_id"] = self.tokenizer.eos_token_id

            # Execute generation
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **params)
                
            # Post-process outputs
            if not self.is_seq2seq:
                # Retrieve only the generated tokens (after the length of the input prompt)
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
            else:
                generated_tokens = outputs[0]
                
            # Decode generated token IDs back to text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during LLM text generation: {e}")
            return f"Error: Unable to generate text due to model failure. Detail: {e}"

    def build_rag_prompt(self, context: str, question: str) -> str:
        """
        Constructs a structured prompt forcing strict context adherence and formatting rules.
        """
        q_lower = question.lower()
        
        # Detect intent
        is_explain = any(kw in q_lower for kw in ["explain", "simple words", "easy explanation", "explain simply"])
        is_summary = any(kw in q_lower for kw in ["summarize", "summary", "about", "brief summary", "main idea"])
        is_key_points = any(kw in q_lower for kw in ["key points", "important points", "bullet points", "list points", "bulleted list", "advantages", "disadvantages", "features", "objectives", "importance"])
        
        if is_explain:
            instruction = (
                "You are a PDF QA assistant.\n"
                "Provide a simple explanation in easy English (maximum 120 words) of the provided context.\n"
                "Do NOT copy long paragraphs verbatim. Rewrite the explanation naturally in your own words.\n"
                "Answer ONLY using the provided context."
            )
        elif is_summary:
            instruction = (
                "You are a PDF QA assistant.\n"
                "Write a concise summary (80 to 150 words maximum) of the provided context, covering only the main topics.\n"
                "Do NOT copy long paragraphs verbatim. Synthesize in your own words.\n"
                "Answer ONLY using the provided context."
            )
        elif is_key_points:
            instruction = (
                "You are a PDF QA assistant.\n"
                "Extract the key points from the provided context (maximum 5 points).\n"
                "Format each key point as a short, clear, and concise item. Do not paste whole paragraphs.\n"
                "Answer ONLY using the provided context."
            )
        else:
            instruction = (
                "You are a PDF QA assistant.\n"
                "Answer the question briefly in a clean, natural sentence.\n"
                "Do NOT copy large paragraphs or whole sentences verbatim. Rewrite naturally.\n"
                "Answer ONLY from the provided context.\n"
                "If the answer is not contained in the context, reply:\n"
                "'I couldn't find this information in the uploaded document.'\n"
                "Do not use outside knowledge."
            )
            
        prompt = (
            f"{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
        return prompt
