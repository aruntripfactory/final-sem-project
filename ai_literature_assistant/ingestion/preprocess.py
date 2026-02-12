# ingestion/preprocess.py
import re
import logging
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Dataclass for structured paper metadata."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    journal_conference: str = ""
    doi: str = ""
    
@dataclass
class TextChunk:
    """Dataclass for structured text chunks."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str = ""

class ResearchPaperChunker:
    """Enhanced chunker for academic papers with better pattern matching."""
    
    # Regex patterns for academic paper components
    TITLE_PATTERNS = [
        r'^[A-Z][A-Za-z\s:,-]{10,200}$',  # Title-like lines
        r'^(?:Title|TITLE)[:\s]+(.+)$',  # Explicit title markers
    ]
    
    AUTHOR_PATTERNS = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',  # Multiple authors
        r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # Middle initial format
        r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+et\s+al\.)',  # et al. format
    ]
    
    SECTION_HEADERS = {
        'abstract': [r'^Abstract', r'^ABSTRACT', r'^Summary'],
        'introduction': [r'^1\.', r'^Introduction', r'^INTRODUCTION'],
        'methodology': [r'^2\.', r'^Methodology', r'^Methods', r'^METHOD'],
        'results': [r'^3\.', r'^Results', r'^RESULTS', r'^Findings'],
        'discussion': [r'^4\.', r'^Discussion', r'^DISCUSSION'],
        'conclusion': [r'^5\.', r'^Conclusion', r'^CONCLUSIONS'],
        'references': [r'^References', r'^Bibliography', r'^REFERENCES'],
    }
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )
    
    def extract_metadata(self, text: str) -> PaperMetadata:
        """Extract structured metadata from paper text."""
        metadata = PaperMetadata()
        lines = text.split('\n')
        
        # Extract title
        metadata.title = self._extract_title(lines)
        
        # Extract authors and affiliations
        metadata.authors, metadata.affiliations = self._extract_authors_and_affiliations(lines)
        
        # Extract abstract
        metadata.abstract = self._extract_abstract(lines)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(lines)
        
        # Extract publication year
        metadata.publication_year = self._extract_publication_year(lines)
        
        return metadata
    
    def _extract_title(self, lines: List[str]) -> str:
        """
        Robust title extraction for academic PDFs.
        Strategy:
        - Look only at top part of document
        - Ignore empty, numeric, conference/journal lines
        - Stop before abstract/authors/intro
        - Use better heuristics for title detection
        - Handle titles split across multiple lines or with missing spaces
        """
        title_candidates = []
        in_title = False
        title_complete = False
        
        print("[TITLE EXTRACTION] Starting title extraction...")
        print(f"[TITLE EXTRACTION] First 10 lines preview:")
        for i, line in enumerate(lines[:10]):
            print(f"  Line {i}: [{len(line.strip())}] {line[:100]}")

        for i, line in enumerate(lines[:40]):  # Increased from 30 to 40
            clean = line.strip()

            if not clean or len(clean) < 3:
                # Empty lines might indicate end of title
                if in_title and title_candidates and len(" ".join(title_candidates)) > 20:
                    title_complete = True
                    print(f"[TITLE EXTRACTION] Title complete (empty line after substantial content)")
                    break
                continue

            # Stop when body starts
            if re.search(r'^(abstract|authors?|introduction|keywords?)\b', clean, re.IGNORECASE):
                print(f"[TITLE EXTRACTION] Stopped at section marker: {clean[:50]}")
                break
            
            # Stop at author names (contains numbers like 1,2 and symbols like ∗)
            if re.search(r'[0-9].*[,∗]|[∗].*[0-9]', clean) and i < 10:
                print(f"[TITLE EXTRACTION] Stopped at author line: {clean[:50]}")
                break

            # Skip header noise (page numbers, conference info, etc.)
            if i < 15:  # Only skip these in first few lines
                # Skip page numbers, years alone on a line
                if re.match(r'^\d+$', clean) or re.match(r'^(19|20)\d{2}$', clean):
                    print(f"[TITLE EXTRACTION] Skipping page/year: {clean}")
                    continue
                # Skip common header patterns
                if re.search(r'^(proceedings|conference|journal|volume|issue|doi:|arxiv:|pages?:)', clean, re.IGNORECASE):
                    print(f"[TITLE EXTRACTION] Skipping header: {clean[:50]}")
                    continue
                # Skip publisher/venue info
                if re.search(r'(ieee|springer|acm|elsevier|wiley|nature|science)\s+(transactions|proceedings|journal)', clean, re.IGNORECASE):
                    print(f"[TITLE EXTRACTION] Skipping publisher: {clean[:50]}")
                    continue
                # Skip copyright and publication info
                if re.search(r'(copyright|©|\(c\)|published|received|accepted)', clean, re.IGNORECASE):
                    print(f"[TITLE EXTRACTION] Skipping copyright: {clean[:50]}")
                    continue

            # Skip lines with mostly special characters or numbers
            if re.match(r'^[\d\W]{5,}$', clean):
                continue

            # Skip URLs and emails
            if re.search(r'(http://|https://|www\.|@.+\..+)', clean, re.IGNORECASE):
                continue

            # Potential title characteristics:
            # - Starts with capital letter
            # - Contains at least 2 words
            # - Length between 10-200 characters
            # - Not all caps (unless short)
            word_count = len(clean.split())
            
            # Good title candidate criteria
            # Special handling for all-caps titles (common in academic papers)
            is_all_caps_title = (
                clean.isupper() and
                10 <= len(clean) <= 200 and  # Lowered from 15 to 10
                word_count >= 1 and  # Allow single words
                i < 5  # Only consider all-caps in first 5 lines as potential title
            )
            
            # If we're already collecting title and this is also all-caps in first 3 lines, include it
            continuing_title = (
                in_title and
                i < 3 and
                clean.isupper() and
                len(clean) >= 10 and
                word_count >= 1
            )
            
            is_good_candidate = (
                10 <= len(clean) <= 200 and
                word_count >= 2 and
                clean[0].isupper() and
                not (clean.isupper() and len(clean) > 50 and i >= 5)  # Avoid all-caps headers later in doc
            ) or is_all_caps_title or continuing_title

            if is_good_candidate:
                in_title = True
                title_candidates.append(clean)
                print(f"[TITLE EXTRACTION] Added candidate {len(title_candidates)}: {clean[:60]}")
                
                # Continue collecting if we're still in all-caps title mode (first few lines)
                if i < 3 and clean.isupper() and len(" ".join(title_candidates)) < 150:
                    print(f"[TITLE EXTRACTION] Continuing to collect all-caps title...")
                    continue  # Keep collecting title parts
                
                # If we have a substantial title and hit an empty line or punctuation end, stop
                if len(" ".join(title_candidates)) > 15:
                    # Check if this line ends with proper punctuation (title ending)
                    if clean.endswith(('.', '!', '?', ':')):
                        title_complete = True
                        print(f"[TITLE EXTRACTION] Title complete (punctuation)")
                        break
                    # Or if next line is empty/different, we likely have complete title
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Stop if next line is empty or starts with lowercase (author names) or has @ (email)
                        if not next_line or (next_line and next_line[0].islower()) or '@' in next_line:
                            title_complete = True
                            print(f"[TITLE EXTRACTION] Title complete (empty/author line)")
                            break
                        
            elif in_title and not is_good_candidate:
                # We were collecting title but now hit non-title content
                print(f"[TITLE EXTRACTION] Stopped at non-title content: {clean[:50]}")
                break

            # Stop if title is getting too long
            if len(" ".join(title_candidates)) > 250:
                print(f"[TITLE EXTRACTION] Title too long, stopping")
                break

        # Join and clean the title
        title = " ".join(title_candidates).strip()
        
        # Final cleanup
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        title = re.sub(r'^[\W\d]+', '', title)  # Remove leading special chars/numbers
        title = re.sub(r'[\W\d]+$', '', title)  # Remove trailing special chars/numbers
        
        # Fix common PDF extraction issues where words are joined
        # Insert space before capital letters that follow lowercase (e.g., "testGap" -> "test Gap")
        title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
        # Insert space when lowercase is followed by uppercase word (e.g., "theTRAIN" -> "the TRAIN")
        title = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', title)
        
        # If title is all caps, convert to title case for readability
        if title and title.isupper() and len(title) > 20:
            # Convert all caps to title case, but preserve acronyms
            words = title.split()
            title_cased = []
            for word in words:
                # Keep short words (likely acronyms) in uppercase
                if len(word) <= 3 and word.isupper():
                    title_cased.append(word)
                else:
                    title_cased.append(word.capitalize())
            title = ' '.join(title_cased)
        
        final_title = title if title and len(title) >= 10 else ""
        print(f"[TITLE EXTRACTION] Final title: {final_title}")
        
        return final_title

    def _extract_authors_and_affiliations(self, lines: List[str]) -> tuple[List[str], List[str]]:
        """Extract authors and their affiliations."""
        authors = []
        affiliations = []
        
        # Look for author section (usually after title)
        for i in range(min(20, len(lines))):
            line = lines[i].strip()
            
            # Check for author patterns
            for pattern in self.AUTHOR_PATTERNS:
                matches = re.findall(pattern, line)
                if matches:
                    authors.extend(matches)
            
            # Look for affiliations
            if any(keyword in line.lower() for keyword in ['university', 'institute', 'college', 'lab']):
                affiliations.append(line)
            
            # Common delimiter for author section end
            if 'abstract' in line.lower() or '1.' in line:
                break
        
        return list(set(authors)), list(set(affiliations))
    
    def _extract_abstract(self, lines: List[str]) -> str:
        """Extract abstract section."""
        abstract_lines = []
        in_abstract = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Start of abstract
            if re.match(r'^(Abstract|ABSTRACT|Summary)', line_stripped, re.IGNORECASE):
                in_abstract = True
                continue
            
            # End of abstract (next section or keywords)
            if in_abstract:
                if (re.match(r'^(Keywords|KEYWORDS|1\.|Introduction)', line_stripped, re.IGNORECASE) or
                    len(abstract_lines) > 500):  # Limit length
                    break
                if line_stripped:
                    abstract_lines.append(line_stripped)
        
        return ' '.join(abstract_lines)
    
    def _extract_keywords(self, lines: List[str]) -> List[str]:
        """Extract keywords section."""
        for i, line in enumerate(lines):
            if re.search(r'Keywords?[:\s]', line, re.IGNORECASE):
                # Extract text after "Keywords:"
                keyword_text = re.split(r'Keywords?[:\s]+', line, flags=re.IGNORECASE)[-1]
                # Split by common delimiters
                keywords = re.split(r'[;,]', keyword_text)
                return [k.strip() for k in keywords if k.strip()]
        return []
    
    def _extract_publication_year(self, lines: List[str]) -> Optional[int]:
        """Extract publication year."""
        for line in lines[:50]:
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            if year_match:
                try:
                    return int(year_match.group())
                except ValueError:
                    continue
        return None
    
    def chunk_document(self, doc_data: Dict[str, Any]) -> List[TextChunk]:
        """
        Create structured chunks from a document dictionary.
        
        Args:
            doc_data: Dictionary containing 'content' and potentially 'title', 'paper_id'
            
        Returns:
            List of TextChunk objects
        """
        text = doc_data.get('content', '')
        # Try to get paper_id, fall back to title, then empty string
        paper_id = doc_data.get('paper_id', doc_data.get('title', ""))
        return self.chunk_text(text, paper_id)

    def chunk_text(self, text: str, paper_id: str = "") -> List[TextChunk]:
        """
        Create structured chunks from paper text.
        
        Args:
            text: Full text of the research paper
            paper_id: Identifier for the paper
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Extract metadata
        metadata = self.extract_metadata(text)
        
        # Create metadata chunks
        chunks.extend(self._create_metadata_chunks(metadata, paper_id))
        
        # Create content chunks by sections
        content_chunks = self._create_content_chunks(text, paper_id)
        chunks.extend(content_chunks)
        
        # Assign unique chunk IDs
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                # Use paper_id + index if available, else static prefix + uuid
                prefix = paper_id if paper_id else "chunk"
                # Clean prefix for valid ID (no spaces/special chars)
                clean_prefix = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', prefix)
                chunk.chunk_id = f"{clean_prefix}_{i}"
        
        return chunks
    
    def _create_metadata_chunks(self, metadata: PaperMetadata, paper_id: str) -> List[TextChunk]:
        """Create chunks for paper metadata."""
        metadata_chunks = []

        # Title chunk (always store, even fallback)
        title_text = metadata.title if metadata.title else "Title not detected from PDF"

        # Store title in multiple formats for better retrieval
        metadata_chunks.append(TextChunk(
            text=f"Title: {title_text}",
            metadata={
                "paper_id": paper_id,
                "chunk_type": "title",
                "is_metadata": True,
                "section": "metadata",
                "importance": "high",
                "title": title_text  # Store title in metadata too
            }
        ))
        
        # Add a separate chunk with just the title for direct matching
        if metadata.title:
            metadata_chunks.append(TextChunk(
                text=f"The title of this paper is: {title_text}. Paper title: {title_text}",
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "title_reference",
                    "is_metadata": True,
                    "section": "metadata",
                    "importance": "high",
                    "title": title_text
                }
            ))

        
        # Authors chunk
        if metadata.authors:
            metadata_chunks.append(TextChunk(
                text=f"Authors: {', '.join(metadata.authors)}",
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "authors",
                    "is_metadata": True,
                    "section": "metadata",
                    "author_count": len(metadata.authors)
                }
            ))
        
        # Abstract chunk (split if too long)
        if metadata.abstract:
            abstract_chunks = self.text_splitter.split_text(metadata.abstract)
            for i, chunk in enumerate(abstract_chunks):
                metadata_chunks.append(TextChunk(
                    text=f"Abstract (Part {i+1}/{len(abstract_chunks)}): {chunk}",
                    metadata={
                        "paper_id": paper_id,
                        "chunk_type": "abstract",
                        "is_metadata": True,
                        "section": "metadata",
                        "part": i+1,
                        "total_parts": len(abstract_chunks)
                    }
                ))
        
        return metadata_chunks
    
    def _create_content_chunks(self, text: str, paper_id: str) -> List[TextChunk]:
        """Create chunks for paper content by sections."""
        chunks = []
        lines = text.split('\n')
        
        current_section = "introduction"
        current_content = []
        section_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for section headers
            detected_section = self._detect_section(line_stripped)
            
            if detected_section and i > section_start + 5:  # Minimum content length
                # Save current section
                if current_content:
                    section_text = '\n'.join(current_content)
                    section_chunks = self._split_section_text(section_text, paper_id, current_section)
                    chunks.extend(section_chunks)
                
                # Start new section
                current_section = detected_section
                current_content = [line_stripped]
                section_start = i
            else:
                current_content.append(line_stripped)
        
        # Add final section
        if current_content:
            section_text = '\n'.join(current_content)
            section_chunks = self._split_section_text(section_text, paper_id, current_section)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _detect_section(self, line: str) -> Optional[str]:
        """Detect if a line is a section header."""
        for section_name, patterns in self.SECTION_HEADERS.items():
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return section_name
        return None
    
    def _split_section_text(self, text: str, paper_id: str, section_name: str) -> List[TextChunk]:
        """Split section text into appropriately sized chunks."""
        chunks = []
        text_chunks = self.text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(TextChunk(
                text=chunk_text,
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "content",
                    "is_metadata": False,
                    "section": section_name,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            ))
        
        return chunks

# For backward compatibility
def clean_text(text: str) -> str:
    """Basic text cleaning for preprocessing."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Legacy function for simple chunking."""
    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(text)
    return [chunk.text for chunk in chunks]

def chunk_text_with_metadata(text: str, paper_id: str = "") -> List[Dict[str, Any]]:
    """Legacy function returning dictionary format."""
    chunker = ResearchPaperChunker()
    chunks = chunker.chunk_text(text, paper_id)
    return [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]