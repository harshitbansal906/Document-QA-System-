import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks recursively using a list of separators.
    Attempts to split by the first separator, and if the chunks are still too large,
    moves to the next separator. Once the text is split into small units, it merges
    them into chunks of size `chunk_size` with `chunk_overlap` overlap.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, chunk_overlap)
        # Ensure chunk_overlap is strictly less than chunk_size
        if self.chunk_overlap >= chunk_size:
            logger.warning(
                f"chunk_overlap ({chunk_overlap}) is greater than or equal to chunk_size ({chunk_size}). "
                f"Setting overlap to chunk_size // 2."
            )
            self.chunk_overlap = chunk_size // 2
            
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits the text using the list of separators."""
        if not text:
            return []
            
        # If the text is already smaller than chunk_size, no need to split further
        if len(text) <= self.chunk_size:
            return [text]
            
        # Get the next separator to use
        if not separators:
            # If no separators left, force split by length
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
        separator = separators[0]
        next_separators = separators[1:]
        
        # Split text by current separator
        if separator == "":
            # Character-by-character split
            splits = list(text)
        else:
            splits = text.split(separator)
            
        # Reconstruct with the separator
        final_splits = []
        for i, split in enumerate(splits):
            # Re-add separator where appropriate
            if i < len(splits) - 1 and separator != "":
                split_with_sep = split + separator
            else:
                split_with_sep = split
                
            if len(split_with_sep) <= self.chunk_size:
                final_splits.append(split_with_sep)
            else:
                # Recurse if the split is still too large
                recursive_splits = self._split_text_recursive(split_with_sep, next_separators)
                final_splits.extend(recursive_splits)
                
        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merges small text splits into chunks of target chunk_size with chunk_overlap."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            # If a single split is larger than chunk_size (should rarely happen after recursion),
            # add it as its own chunk
            if split_len > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(split)
                continue
                
            if current_length + split_len > self.chunk_size:
                # Save the current chunk
                joined_chunk = "".join(current_chunk)
                chunks.append(joined_chunk)
                
                # Start a new chunk, keeping the overlap
                overlap_chars = []
                overlap_len = 0
                
                # Look backward in current_chunk to extract overlap
                for piece in reversed(current_chunk):
                    if overlap_len + len(piece) <= self.chunk_overlap:
                        overlap_chars.insert(0, piece)
                        overlap_len += len(piece)
                    else:
                        # Grab partial piece if needed
                        remaining = self.chunk_overlap - overlap_len
                        if remaining > 0:
                            overlap_chars.insert(0, piece[-remaining:])
                        break
                        
                current_chunk = overlap_chars + [split]
                current_length = sum(len(c) for c in current_chunk)
            else:
                current_chunk.append(split)
                current_length += split_len
                
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks

    def split_text(self, text: str) -> List[str]:
        """Splits a single text block into chunks."""
        splits = self._split_text_recursive(text, self.separators)
        # Filter out empty splits before merging
        clean_splits = [s for s in splits if s.strip()]
        return self._merge_splits(clean_splits)


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Takes list of page dicts and splits them into chunks with metadata.
    Filters out any empty chunks.
    
    Prints logs to stdout as required:
      - Chunks Created: <count>
      - Average Chunk Length: <avg_length>
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        if not text:
            continue
            
        chunks = splitter.split_text(text)
        
        for i, chunk_text in enumerate(chunks):
            # Strict cleaning of whitespace
            clean_chunk = chunk_text.strip()
            # Remove/skip empty chunks
            if not clean_chunk:
                continue
                
            chunk_metadata = metadata.copy()
            # Generate a unique chunk identifier
            source_name = metadata.get("source", "unknown")
            page_num = metadata.get("page", 0)
            chunk_metadata["chunk_id"] = f"{source_name}_p{page_num}_c{i}"
            
            all_chunks.append({
                "text": clean_chunk,
                "metadata": chunk_metadata
            })
            
    # Calculate average chunk length
    if all_chunks:
        avg_chunk_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    else:
        avg_chunk_len = 0.0

    # Print logs as required
    print(f"Chunks Created: {len(all_chunks)}")
    print(f"Average Chunk Length: {avg_chunk_len:.2f}")

    logger.info(f"Chunks Created: {len(all_chunks)}")
    logger.info(f"Average Chunk Length: {avg_chunk_len:.2f}")
    
    return all_chunks
