"""
Script to index all documents from downloaded_documents folder
Scans the folder, extracts text, uploads to CosmosDB, and indexes with GraphRAG
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add parent directory to path to import app modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import existing functions
from app.cosmos_crud import (
    get_firm_managers,
    extract_text_from_docx,
    extract_text_from_xlsx,
    clean_ocr_text,
    index_documents_for_case
)
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from docx import Document as DocxDocument

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DOWNLOADED_DOCUMENTS_FOLDER = Path("downloaded_documents")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-25.07.0\Library\bin")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR")

pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, "tesseract.exe")

# Default values
DEFAULT_FIRM_ID = "1680"
DEFAULT_CASE_ID = "29493"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.txt': 'text',
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.xlsx': 'xlsx',
    '.xls': 'xlsx',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.gif': 'image',
    '.bmp': 'image',
    '.tiff': 'image',
    '.tif': 'image',
    '.JPG': 'image',
    '.JPEG': 'image',
    '.PNG': 'image',
}


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    file_extension = file_path.suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(file_extension, None)
    
    if not file_type:
        logger.warning(f"Unsupported file type: {file_extension} for {file_path.name}")
        return ""
    
    try:
        if file_type == 'text':
            # Plain text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif file_type == 'pdf':
            # PDF file - try text extraction first, then OCR
            try:
                reader = PdfReader(str(file_path))
                text = "".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PDF text extraction failed for {file_path.name}: {e}")
            
            # Fallback to OCR
            try:
                logger.info(f"Running OCR on {file_path.name}...")
                images = convert_from_path(str(file_path), poppler_path=POPPLER_PATH)
                ocr_text = []
                for img in images:
                    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
                    ocr_text.append(text)
                return "\n".join(ocr_text)
            except Exception as e:
                logger.error(f"OCR failed for {file_path.name}: {e}")
                return ""
        
        elif file_type == 'docx':
            # DOCX file
            return extract_text_from_docx(str(file_path))
        
        elif file_type == 'xlsx':
            # XLSX file
            return extract_text_from_xlsx(str(file_path))
        
        elif file_type == 'image':
            # Image file - use OCR
            try:
                logger.info(f"Running OCR on image {file_path.name}...")
                from PIL import Image
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
                return text
            except Exception as e:
                logger.error(f"OCR failed for image {file_path.name}: {e}")
                return ""
        
    except Exception as e:
        logger.error(f"Error extracting text from {file_path.name}: {e}")
        return ""
    
    return ""


async def process_document(
    file_path: Path,
    case_id: str,
    firm_id: str,
    document_counter: int
) -> Dict[str, Any]:
    """
    Process a single document: extract text and store in CosmosDB.
    
    Args:
        file_path: Path to the document file
        case_id: Case identifier
        firm_id: Firm identifier
        document_counter: Sequential document ID
        
    Returns:
        Dictionary with processing results
    """
    result = {
        'filename': file_path.name,
        'path': str(file_path),
        'success': False,
        'error': None,
        'text_length': 0,
        'document_id': document_counter
    }
    
    try:
        logger.info(f"[{document_counter}] Processing: {file_path.name}")
        
        # Extract text
        extracted_text = extract_text_from_file(file_path)
        
        if not extracted_text.strip():
            result['error'] = "No text extracted"
            logger.warning(f"[{document_counter}] No text extracted from {file_path.name}")
            return result
        
        # Clean text
        extracted_text = clean_ocr_text(extracted_text)
        result['text_length'] = len(extracted_text)
        
        # Get firm-specific managers
        firm_managers = get_firm_managers(firm_id, case_id)
        firm_input_manager = firm_managers['input']
        
        # Create filename for storage
        file_extension = file_path.suffix.lower()
        stored_filename = f"{firm_id}_{case_id}_{document_counter}.txt"
        
        # Store extracted text in CosmosDB
        stored_doc = firm_input_manager.store_extracted_text(
            case_id=case_id,
            document_id=document_counter,
            text_content=extracted_text,
            filename=stored_filename,
            original_filename=file_path.name,
            firm_id=firm_id
        )
        
        if stored_doc:
            result['success'] = True
            if "part_number" in stored_doc and "total_parts" in stored_doc:
                total_parts = stored_doc.get("total_parts", 1)
                logger.info(f"[{document_counter}] ✓ Saved in {total_parts} parts: {file_path.name} ({len(extracted_text)} chars)")
            else:
                logger.info(f"[{document_counter}] ✓ Saved: {file_path.name} ({len(extracted_text)} chars)")
        else:
            result['error'] = "Failed to store document"
            logger.error(f"[{document_counter}] ✗ Failed to store: {file_path.name}")
    
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[{document_counter}] ✗ Error processing {file_path.name}: {e}")
    
    return result


async def index_all_documents_from_folder(
    folder_path: Path,
    case_id: str,
    firm_id: str,
    run_indexing: bool = True
) -> Dict[str, Any]:
    """
    Index all documents from a folder.
    
    Args:
        folder_path: Path to the folder containing documents
        case_id: Case identifier
        firm_id: Firm identifier
        run_indexing: Whether to run GraphRAG indexing after uploading documents
        
    Returns:
        Dictionary with processing statistics
    """
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    logger.info(f"Starting batch indexing from: {folder_path}")
    logger.info(f"Case ID: {case_id}, Firm ID: {firm_id}")
    
    # Get all supported files
    all_files = []
    for ext in SUPPORTED_EXTENSIONS.keys():
        all_files.extend(folder_path.rglob(f"*{ext}"))
        all_files.extend(folder_path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    all_files = sorted(set(all_files))
    
    if not all_files:
        logger.warning(f"No supported files found in {folder_path}")
        return {
            'total_files': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'results': []
        }
    
    logger.info(f"Found {len(all_files)} files to process")
    
    # Get firm-specific managers to initialize containers
    firm_managers = get_firm_managers(firm_id, case_id)
    firm_input_manager = firm_managers['input']
    
    # Process documents sequentially (to avoid overwhelming CosmosDB)
    results = []
    document_counter = 1
    
    for file_path in all_files:
        result = await process_document(file_path, case_id, firm_id, document_counter)
        results.append(result)
        
        if result['success']:
            document_counter += 1
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_text_chars = sum(r['text_length'] for r in results if r['success'])
    
    stats = {
        'total_files': len(all_files),
        'processed': len(results),
        'successful': successful,
        'failed': failed,
        'total_text_chars': total_text_chars,
        'results': results
    }
    
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files found: {stats['total_files']}")
    logger.info(f"Successfully processed: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total text characters: {stats['total_text_chars']:,}")
    logger.info("=" * 60)
    
    # Run indexing if requested
    if run_indexing and successful > 0:
        logger.info("Starting GraphRAG indexing...")
        try:
            await index_documents_for_case(case_id, firm_id)
            logger.info("✓ Indexing completed successfully")
            stats['indexing_success'] = True
        except Exception as e:
            logger.error(f"✗ Indexing failed: {e}")
            stats['indexing_success'] = False
            stats['indexing_error'] = str(e)
    else:
        stats['indexing_success'] = None
        logger.info("Skipping indexing (run_indexing=False or no successful uploads)")
    
    return stats


async def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Index all documents from downloaded_documents folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="downloaded_documents",
        help="Path to folder containing documents (default: downloaded_documents)"
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=DEFAULT_CASE_ID,
        help=f"Case ID for indexing (default: {DEFAULT_CASE_ID})"
    )
    parser.add_argument(
        "--firm-id",
        type=str,
        default=DEFAULT_FIRM_ID,
        help=f"Firm ID for indexing (default: {DEFAULT_FIRM_ID})"
    )
    parser.add_argument(
        "--no-indexing",
        action="store_true",
        help="Skip GraphRAG indexing (only upload documents)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="Specific file extensions to process (e.g., --extensions .pdf .docx)"
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    
    # Override supported extensions if specified
    if args.extensions:
        global SUPPORTED_EXTENSIONS
        SUPPORTED_EXTENSIONS = {
            ext: SUPPORTED_EXTENSIONS.get(ext, 'text')
            for ext in args.extensions
        }
        logger.info(f"Processing only: {', '.join(args.extensions)}")
    
    # Run the indexing
    stats = await index_all_documents_from_folder(
        folder_path=folder_path,
        case_id=args.case_id,
        firm_id=args.firm_id,
        run_indexing=not args.no_indexing
    )
    
    # Print failed files if any
    failed_files = [r for r in stats['results'] if not r['success']]
    if failed_files:
        logger.warning("\nFailed files:")
        for r in failed_files:
            logger.warning(f"  - {r['filename']}: {r['error']}")
    
    return stats


if __name__ == "__main__":
    asyncio.run(main())

