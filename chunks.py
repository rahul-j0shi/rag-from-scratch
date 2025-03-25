import json
import os
import textwrap
import tiktoken
import uuid
import logging

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

CHUNK_SIZE = 750

encoding = tiktoken.encoding_for_model(os.environ.get('EMBEDDING_MODEL'))

METADATA_MARK = '---'


def gather_markdown_documents() -> list[str]:
    """Collect all Markdown files from the handbook directory."""
    markdown_files = []
    for directory, _, files in os.walk('handbook'):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(directory, file)
                markdown_files.append(full_path)
    return markdown_files


def extract_document_metadata(document_text: str) -> tuple:
    """Extract metadata from Markdown file."""
    # Find the first two --- sections
    metadata_sections = document_text.split(METADATA_MARK)
    
    # Ensure we have at least two sections (metadata and content)
    if len(metadata_sections) < 3:
        logger.warning("No metadata found in document")
        return "", "", document_text

    # Parse metadata
    metadata = metadata_sections[1].strip()
    title = ""
    description = ""

    for line in metadata.split('\n'):
        line = line.strip()
        if line.startswith('title:'):
            title = line.replace('title:', '').replace('"', '').strip()
        elif line.startswith('description:'):
            description = line.replace('description:', '').replace('"', '').strip()

    # Content is everything after the second --- 
    content = METADATA_MARK.join(metadata_sections[2:]).strip()

    return title, description, content


def create_chunk_file(title: str, description: str, document: str, chunk_index: int, chunk: str) -> None:
    """Create a JSON file for each document chunk."""
    # Ensure chunks directory exists
    os.makedirs('chunks', exist_ok=True)

    chunk_id = str(uuid.uuid4())
    filename = os.path.basename(document).replace('.md', f'-{chunk_index}.json')
    
    with open(os.path.join('chunks', filename), 'w', encoding='utf-8') as chunk_file:
        json.dump({
            'id': chunk_id,
            'title': title,
            'description': description,
            'document': document,
            'chunk_text': chunk,
            'chunk_token_count': len(encoding.encode(chunk))
        }, chunk_file, indent=4, ensure_ascii=False)


def chunk_markdown_documents():
    """Chunk Markdown documents into smaller pieces."""
    documents = gather_markdown_documents()
    logger.info(f"Found {len(documents)} Markdown files to process")

    for document in documents:
        try:
            # Read file with UTF-8 encoding
            with open(document, 'r', encoding='utf-8') as d:
                document_text = d.read()

            # Extract metadata
            title, description, remaining_text = extract_document_metadata(document_text)
            logger.info(f"Processing document: {document}")

            # Chunk the text
            chunks = []
            words = remaining_text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                # If adding this word would exceed chunk size, start a new chunk
                if current_length + len(word) + (1 if current_length > 0 else 0) > CHUNK_SIZE:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                current_chunk.append(word)
                current_length += len(word) + (1 if current_length > 0 else 0)

            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Create chunk files
            for chunk_index, chunk in enumerate(chunks, start=1):
                create_chunk_file(title, description, document, chunk_index, chunk)

        except Exception as e:
            logger.error(f"Error processing {document}: {e}")


if __name__ == '__main__':
    chunk_markdown_documents()