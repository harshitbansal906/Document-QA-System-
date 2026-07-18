import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from utils.pdf_loader import load_multiple_pdfs
from utils.text_splitter import split_documents
from utils.embedder import LocalEmbedder
from utils.vector_store import LocalVectorStore
from utils.llm import LocalLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Unified Orchestrator linking document ingestion, text splitting, 
    embedding generation, FAISS indexing, similarity retrieval, and local LLM generation.
    Supports lazy loading of heavy models (embedding and LLM) to conserve memory.
    """
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        models_dir: str = None,
        vector_store_dir: str = None
    ):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.models_dir = models_dir
        
        # Instantiate VectorStore shell (very lightweight, does not load models)
        self.vector_store = LocalVectorStore(store_dir=vector_store_dir)
        
        # Lazy model initializers
        self._embedder = None
        self._llm = None

    @property
    def embedder(self) -> LocalEmbedder:
        """Lazy-loaded local embedding model."""
        if self._embedder is None:
            logger.info("Lazy-loading local embedder model...")
            self._embedder = LocalEmbedder(
                model_name=self.embedding_model_name,
                models_dir=self.models_dir
            )
        return self._embedder

    @property
    def llm(self) -> LocalLLM:
        """Lazy-loaded local HuggingFace LLM model."""
        if self._llm is None:
            logger.info("Lazy-loading local LLM model...")
            self._llm = LocalLLM(
                model_id=self.llm_model_name,
                models_dir=self.models_dir
            )
        return self._llm

    def ingest_files(
        self,
        files: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Runs the complete document ingestion pipeline.
        Extracts, chunks, embeds, indexes, and persists the files.
        
        Args:
            files: List of dicts, e.g. [{"source": file_obj/bytes/path, "name": "filename.pdf"}, ...]
            chunk_size: Size of characters per semantic chunk.
            chunk_overlap: Characters overlapping between consecutive chunks.
            
        Returns:
            Dict containing processing status metrics.
        """
        if not files:
            return {"status": "error", "message": "No files provided for ingestion."}
            
        try:
            logger.info("Step 1: Extracting text from PDF sources...")
            pages = load_multiple_pdfs(files)
            if not pages:
                return {
                    "status": "error",
                    "message": "Failed to extract text from any page. Check if PDFs are corrupted or blank."
                }
                
            logger.info("Step 2: Splitting page text into semantic chunks...")
            chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                return {
                    "status": "error",
                    "message": "Chunk splitting resulted in 0 chunks. Cannot build index."
                }

            logger.info("Step 3: Generating dense vector embeddings for chunks...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_documents(chunk_texts)

            logger.info("Step 4: Building and storing FAISS index...")
            self.vector_store.build_index(chunks, embeddings)
            self.vector_store.save()
            
            # Count distinct documents ingested
            distinct_docs = list(set(p["metadata"]["source"] for p in pages))
            
            return {
                "status": "success",
                "documents_ingested": len(distinct_docs),
                "total_pages": len(pages),
                "total_chunks": len(chunks),
                "message": f"Successfully ingested {len(distinct_docs)} documents."
            }
        except Exception as e:
            logger.error(f"Failed to ingest files in RAGSystem: {e}")
            return {"status": "error", "message": f"Ingestion pipeline failure: {str(e)}"}

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.0
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs the search-and-generation query pipeline.
        1. Embeds query.
        2. Retrieves top context chunks from FAISS (top_k=5).
        3. Rejects low-confidence chunks based on similarity score.
        4. Constructs strict context prompt.
        5. Calls local LLM to generate response.
        
        Args:
            question: The user query string.
            top_k: Number of reference document chunks to retrieve (defaults to 5).
            temperature: Sampling temperature for local LLM generation.
            
        Returns:
            Tuple of:
            - Generated response text (str)
            - Retrieved reference chunks (list)
        """
        if not question:
            return "Please ask a valid question.", []
            
        # Check if vector store is loaded
        if self.vector_store.is_empty():
            # Attempt to auto-load index from disk
            loaded = self.vector_store.load()
            if not loaded:
                return "The vector database is currently empty. Please upload documents first.", []

        try:
            logger.info(f"Retrieving context for question: '{question}'")
            # 1. Embed query
            query_embedding = self.embedder.embed_query(question)
            
            # 2. Retrieve similar chunks (retrieve at least top_k=5)
            search_k = max(5, top_k)
            retrieved_chunks = self.vector_store.search(query_embedding, top_k=search_k)
            
            # Print similarity scores in logs for debugging
            scores = [chunk.get("score", 0.0) for chunk in retrieved_chunks]
            print(f"Retrieved similarity scores: {scores}")
            logger.info(f"Retrieved similarity scores: {scores}")
            
            if not retrieved_chunks:
                logger.info("No chunks retrieved at all. Returning fallback.")
                return "I couldn't find this information in the uploaded document.", []
                
            # 3. Detect if this is a generic document question (e.g. summary, overview, key points)
            generic_keywords = ["about", "summarize", "summary", "key points", "overview", "main topic", "theme", "abstract", "list points"]
            is_generic_query = any(keyword in question.lower() for keyword in generic_keywords)
            
            # Threshold configurations
            EXTREMELY_POOR_THRESHOLD = 0.15
            SPECIFIC_QUERY_THRESHOLD = 0.20
            
            # 4. Filter chunks based on query intent
            if is_generic_query:
                # For generic document questions, answer using the highest-ranked retrieved chunks directly
                filtered_chunks = retrieved_chunks
            else:
                max_score = max(scores) if scores else 0.0
                
                # If all similarity scores are extremely poor for specific queries, return fallback directly
                if max_score < EXTREMELY_POOR_THRESHOLD:
                    logger.info(f"Max similarity score ({max_score:.4f}) is below extremely poor threshold ({EXTREMELY_POOR_THRESHOLD}). Returning fallback.")
                    return "I couldn't find this information in the uploaded document.", []
                
                # For specific factual queries, filter out low-confidence chunks to prevent hallucination
                filtered_chunks = [chunk for chunk in retrieved_chunks if chunk.get("score", 0.0) >= SPECIFIC_QUERY_THRESHOLD]
                if not filtered_chunks:
                    logger.info("All retrieved chunks fell below specific query threshold. Returning fallback.")
                    return "I couldn't find this information in the uploaded document.", []
                
            # 4. Assemble prompt context from filtered chunks
            context_blocks = []
            for i, chunk in enumerate(filtered_chunks):
                # Add indices and labels
                meta = chunk.get("metadata", {})
                source = meta.get("source", "Unknown")
                page = meta.get("page", "?")
                context_blocks.append(
                    f"--- Source Block {i+1} [Doc: {source}, Page: {page}] ---\n"
                    f"{chunk['text']}"
                )
            context_str = "\n\n".join(context_blocks)
            
            # 5. Generate response
            prompt = self.llm.build_rag_prompt(context=context_str, question=question)
            logger.info("Invoking LLM for question answering...")
            answer = self.llm.generate(prompt, temperature=temperature)
            
            # Post-check: standardise negative responses to prevent hallucination
            lower_ans = answer.lower()
            negative_indicators = [
                "could not find", "cannot answer", "couldn't find", 
                "not present", "not contained", "do not know", "don't know",
                "not mentioned", "no information", "unavailable", "unknown"
            ]
            if any(indicator in lower_ans for indicator in negative_indicators):
                return "I couldn't find this information in the uploaded document.", []
                
            # Classify query intent for output block formatting
            q_lower = question.lower()
            is_explain = any(kw in q_lower for kw in ["explain", "simple words", "easy explanation", "explain simply"])
            is_summary = any(kw in q_lower for kw in ["summarize", "summary", "about", "brief summary", "main idea"])
            is_key_points = any(kw in q_lower for kw in ["key points", "important points", "bullet points", "list points", "bulleted list", "advantages", "disadvantages", "features", "objectives", "importance"])
            
            # Clean duplicate prefix headers generated by LLM
            clean_ans = answer
            prefixes_to_strip = [
                "simple explanation:", "summary:", "key points:", "answer:",
                "simple explanation", "summary", "key points", "answer"
            ]
            for prefix in prefixes_to_strip:
                if clean_ans.lower().startswith(prefix):
                    clean_ans = clean_ans[len(prefix):].strip()
                    if clean_ans.startswith(":"):
                        clean_ans = clean_ans[1:].strip()
                    break
                    
            if is_key_points:
                lines = clean_ans.split("\n")
                new_lines = []
                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    # Strip any common bullet list prefixes
                    for marker in ["-", "*", "•", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]:
                        if line_str.startswith(marker):
                            line_str = line_str[len(marker):].strip()
                            break
                    if line_str:
                        new_lines.append(f"• {line_str}")
                # Format to a maximum of 5 bullet points
                clean_ans = "\n".join(new_lines[:5])
                header = "Key Points:"
            elif is_explain:
                header = "Simple Explanation:"
            elif is_summary:
                header = "Summary:"
            else:
                header = "Answer:"
                
            # Retrieve source metadata for citation block
            top_meta = filtered_chunks[0].get("metadata", {})
            source_name = top_meta.get("source", "Unknown")
            filename = os.path.basename(source_name)
            page_num = top_meta.get("page", "?")
            
            formatted_answer = (
                f"{header}\n"
                f"{clean_ans}\n\n"
                f"Source:\n"
                f"{filename} - Page {page_num}"
            )
            
            return formatted_answer, filtered_chunks
            
        except Exception as e:
            logger.error(f"Query execution failure in RAGSystem: {e}")
            return f"An error occurred during search: {e}", []

    def reset_database(self) -> None:
        """Clears vectors and saved indices from memory and storage."""
        self.vector_store.clear()
