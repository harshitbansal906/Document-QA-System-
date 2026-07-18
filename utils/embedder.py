import os
import logging
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Wrapper for local text embedding generation using SentenceTransformers.
    Saves/caches the model files locally to ensure offline usability after initial download.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        models_dir: str = None
    ):
        # Determine the models directory
        if models_dir is None:
            # Resolve to root-level 'models/embeddings'
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_dir = os.path.join(base_dir, "models", "embeddings")
        else:
            self.models_dir = os.path.abspath(models_dir)
            
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Detect compute device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedding device: {self.device}")
        
        # Load / Download model locally
        logger.info(f"Initializing embedding model '{model_name}' cached in '{self.models_dir}'...")
        try:
            # cache_folder is where SentenceTransformer downloads and loads the model
            self.model = SentenceTransformer(
                model_name,
                cache_folder=self.models_dir,
                device=self.device
            )
            # Print exact log required by user
            print("Embedding Model Loaded")
            logger.info("Embedding Model Loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of document strings.
        
        Args:
            texts: List of text blocks to embed.
            
        Returns:
            A numpy array of shape (num_texts, embedding_dimension).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
            
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            )
            # Print logs required by user
            print("Embeddings Generated")
            print(f"Embedding Dimension: {embeddings.shape[1]}")
            
            logger.info("Embeddings Generated")
            logger.info(f"Embedding Dimension: {embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise e

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates embedding for a single user query string.
        
        Args:
            query: The search query string.
            
        Returns:
            A 1D numpy array representing the query embedding vector.
        """
        if not query:
            return np.empty((0,), dtype=np.float32)
            
        try:
            embedding = self.model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise e
