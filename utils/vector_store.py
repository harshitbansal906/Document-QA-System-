import os
import pickle
import logging
from typing import List, Dict, Any
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalVectorStore:
    """
    Manages a local FAISS index alongside chunk metadata serialized via pickle.
    Guarantees persistence to disk and allows fast semantic searches over document vectors.
    """
    def __init__(self, store_dir: str = None):
        # Determine paths
        if store_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.store_dir = os.path.join(base_dir, "vector_store")
        else:
            self.store_dir = os.path.abspath(store_dir)
            
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "faiss.index")
        self.metadata_path = os.path.join(self.store_dir, "chunks_metadata.pkl")
        
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Creates a FAISS index from the given chunk embeddings.
        Normalizes both vectors to perform Cosine Similarity search.
        
        Args:
            chunks: A list of dicts matching embeddings, e.g. [{"text": "...", "metadata": {...}}]
            embeddings: Numpy array of shape (num_chunks, dimension).
        """
        if not chunks or embeddings.size == 0:
            logger.warning("Empty chunks list or embeddings provided. Index creation skipped.")
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of text chunks ({len(chunks)}) "
                f"and number of embeddings ({embeddings.shape[0]})."
            )

        dimension = embeddings.shape[1]
        
        # Clone embeddings to prevent modifying user's array
        normalized_embeddings = embeddings.astype(np.float32).copy()
        # Cosine Similarity is equivalent to Inner Product on L2-normalized vectors
        faiss.normalize_L2(normalized_embeddings)
        
        # IndexFlatIP uses Inner Product
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized_embeddings)
        self.chunks = chunks
        
        # User requested print statements
        print("FAISS Created")
        print(f"Vectors Indexed: {self.index.ntotal}")

        logger.info("FAISS Created")
        logger.info(f"Vectors Indexed: {self.index.ntotal}")

    def save(self) -> None:
        """Saves the FAISS index and chunk metadata files to disk."""
        if self.index is None or not self.chunks:
            logger.error("No active FAISS index exists to save.")
            return

        try:
            # Write FAISS index binary
            faiss.write_index(self.index, self.index_path)
            
            # Serialize text and metadata dicts
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunks, f)
                
            print("Vectors Saved")
            logger.info(f"Successfully saved FAISS index to '{self.index_path}'.")
        except Exception as e:
            logger.error(f"Error saving vector store to disk: {e}")
            raise e

    def load(self) -> bool:
        """
        Loads the index and metadata from disk if they exist.
        
        Returns:
            True if loaded successfully, False if files are missing.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.info("No saved FAISS index or metadata files found on disk.")
            return False

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
                
            logger.info(
                f"Successfully loaded FAISS index with {self.index.ntotal} vectors "
                f"from '{self.index_path}'."
            )
            return True
        except Exception as e:
            logger.error(f"Error loading vector store from disk: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar text chunks for a given query embedding.
        
        Args:
            query_embedding: Numpy vector of query text.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of matching chunks with similarity scores included.
        """
        if self.index is None or not self.chunks:
            logger.warning("Search failed: Vector store is empty or not loaded.")
            return []

        # Ensure correct shape (1, dimension)
        q_vec = query_embedding.astype(np.float32).copy()
        if q_vec.ndim == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        # Normalize the query embedding for Cosine similarity
        faiss.normalize_L2(q_vec)
        
        # Limit top_k to elements present in the database
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        # Search index
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS index outputs -1 if not enough elements exist
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx].copy()
            # Score is cosine similarity value (-1.0 to 1.0)
            chunk["score"] = float(score)
            results.append(chunk)
            
        # User requested print statement
        print(f"Retrieved Chunks: {len(results)}")
        logger.info(f"Retrieved Chunks: {len(results)}")
            
        return results

    def clear(self) -> None:
        """Clears local state and deletes files from disk."""
        self.index = None
        self.chunks = []
        
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception as e:
                logger.warning(f"Could not remove index file: {e}")
                
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
            except Exception as e:
                logger.warning(f"Could not remove metadata file: {e}")
                
        logger.info("Cleared local vector store and deleted state files.")

    def is_empty(self) -> bool:
        """Returns True if no vectors are loaded."""
        return self.index is None or self.index.ntotal == 0
