import os
import io
import logging
from typing import List, Dict, Union, Any
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails due to file corruption or parsing issues."""
    pass

class InvalidPDFInputError(Exception):
    """Raised when the input arguments for PDF extraction are invalid."""
    pass


def extract_multiple_pdfs_combined(
    sources: List[Union[str, io.BytesIO, bytes]],
    filenames: List[str]
) -> str:
    """
    Extracts text from multiple PDF sources, ignoring encrypted PDFs and skipping empty pages,
    and returns a single combined string of all extracted content.
    
    Prints logs to stdout as required:
      - PDF Loaded: <filename>
      - Number of PDFs: <count>
      - Number of Pages: <total_pages>
      - Characters Extracted: <total_chars>
      
    Raises meaningful exceptions for invalid inputs or critical extraction failures.
    """
    if not sources or not filenames:
        raise InvalidPDFInputError("Sources and filenames lists must not be empty.")
        
    if len(sources) != len(filenames):
        raise InvalidPDFInputError(
            f"Mismatch between number of sources ({len(sources)}) and filenames ({len(filenames)})."
        )

    combined_text_parts = []
    total_pages_processed = 0
    loaded_pdfs_count = 0
    total_characters_extracted = 0

    for source, filename in zip(sources, filenames):
        try:
            # Resolve the source to a file-like object
            if isinstance(source, bytes):
                pdf_file = io.BytesIO(source)
            elif isinstance(source, io.BytesIO):
                pdf_file = source
            elif isinstance(source, str):
                if not os.path.exists(source):
                    raise FileNotFoundError(f"PDF file not found at path: {source}")
                pdf_file = open(source, "rb")
            else:
                pdf_file = source

            # Load the reader
            try:
                reader = PyPDF2.PdfReader(pdf_file)
            except Exception as e:
                raise PDFExtractionError(f"Failed to parse PDF structure for '{filename}': {e}")

            # Ignore encrypted PDFs
            if reader.is_encrypted:
                logger.warning(f"Ignoring encrypted PDF: {filename}")
                if isinstance(source, str) and not isinstance(pdf_file, io.BytesIO):
                    pdf_file.close()
                continue

            num_pages = len(reader.pages)
            pdf_char_count = 0
            pdf_pages_added = 0

            # Extract page-by-page
            for page_idx in range(num_pages):
                try:
                    page = reader.pages[page_idx]
                    page_text = page.extract_text() or ""
                    cleaned_text = " ".join(page_text.split()).strip()
                    
                    # Skip empty pages
                    if cleaned_text:
                        combined_text_parts.append(cleaned_text)
                        pdf_char_count += len(cleaned_text)
                        pdf_pages_added += 1
                except Exception as page_err:
                    logger.error(f"Error extracting text from page {page_idx + 1} of '{filename}': {page_err}")

            if isinstance(source, str) and not isinstance(pdf_file, io.BytesIO):
                pdf_file.close()

            # Increment count of successfully processed PDFs
            loaded_pdfs_count += 1
            total_pages_processed += pdf_pages_added
            total_characters_extracted += pdf_char_count
            
            # Print PDF Loaded log for this file
            print(f"PDF Loaded: {filename}")
            logger.info(f"PDF Loaded: {filename}")

        except FileNotFoundError as fnf:
            logger.error(fnf)
            raise fnf
        except PDFExtractionError as pde:
            logger.error(pde)
            raise pde
        except Exception as e:
            logger.error(f"Unexpected error processing '{filename}': {e}")
            raise PDFExtractionError(f"Unexpected extraction error for '{filename}': {e}")

    # Output final summary logs to stdout
    print(f"Number of PDFs: {loaded_pdfs_count}")
    print(f"Number of Pages: {total_pages_processed}")
    print(f"Characters Extracted: {total_characters_extracted}")

    logger.info(f"Number of PDFs: {loaded_pdfs_count}")
    logger.info(f"Number of Pages: {total_pages_processed}")
    logger.info(f"Characters Extracted: {total_characters_extracted}")

    return "\n\n".join(combined_text_parts)


def extract_text_from_pdf(
    file_source: Union[str, io.BytesIO, bytes],
    filename: str
) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file source.
    Skips encrypted PDFs and empty pages.
    """
    pages_data = []
    
    try:
        if isinstance(file_source, bytes):
            pdf_file = io.BytesIO(file_source)
        elif isinstance(file_source, io.BytesIO):
            pdf_file = file_source
        elif isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise FileNotFoundError(f"PDF file not found at path: {file_source}")
            pdf_file = open(file_source, "rb")
        else:
            pdf_file = file_source

        reader = PyPDF2.PdfReader(pdf_file)
        
        # Ignore encrypted PDFs
        if reader.is_encrypted:
            logger.warning(f"Ignoring encrypted PDF: {filename}")
            if isinstance(file_source, str) and not isinstance(pdf_file, io.BytesIO):
                pdf_file.close()
            return []

        total_pages = len(reader.pages)
        
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page = reader.pages[page_idx]
                text = page.extract_text() or ""
                cleaned_text = " ".join(text.split()).strip()
                
                # Skip empty pages
                if cleaned_text:
                    pages_data.append({
                        "text": cleaned_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "total_pages": total_pages
                        }
                    })
            except Exception as page_err:
                logger.error(f"Error extracting text from page {page_num} of '{filename}': {page_err}")
                
        if isinstance(file_source, str) and not isinstance(pdf_file, io.BytesIO):
            pdf_file.close()
            
    except Exception as e:
        logger.error(f"Failed to parse PDF file '{filename}': {e}")
        raise PDFExtractionError(f"Failed to parse PDF '{filename}': {e}")
        
    return pages_data


def load_multiple_pdfs(
    sources: List[Dict[str, Union[str, io.BytesIO, bytes, Any]]]
) -> List[Dict[str, Any]]:
    """
    Helper function to load and combine page text from multiple PDF sources.
    Outputs metrics to stdout.
    """
    all_pages = []
    loaded_pdfs = 0
    total_chars = 0
    
    for src in sources:
        file_source = src.get("source")
        name = src.get("name", "Unknown_Document.pdf")
        if file_source is not None:
            pages = extract_text_from_pdf(file_source, name)
            if pages:
                all_pages.extend(pages)
                loaded_pdfs += 1
                total_chars += sum(len(p["text"]) for p in pages)
                # Print PDF Loaded log
                print(f"PDF Loaded: {name}")
                logger.info(f"PDF Loaded: {name}")

    # Output summary logs
    print(f"Number of PDFs: {loaded_pdfs}")
    print(f"Number of Pages: {len(all_pages)}")
    print(f"Characters Extracted: {total_chars}")
    
    logger.info(f"Number of PDFs: {loaded_pdfs}")
    logger.info(f"Number of Pages: {len(all_pages)}")
    logger.info(f"Characters Extracted: {total_chars}")
    
    return all_pages
