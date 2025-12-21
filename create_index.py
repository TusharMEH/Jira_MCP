"""
IT Support Knowledge Base - PDF Chunking and FAISS Indexing
Reads PDF, chunks by section headings, embeds with sentence-transformers, indexes with FAISS
"""

import os
import re
import pickle
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_PATH = "IT_Support_Knowledge_Base.pdf"
INDEX_PATH = "kb_faiss_index.bin"
CHUNKS_PATH = "kb_chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# STEP 1: Extract Text from PDF
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF file"""
    print(f"\n📄 Reading PDF: {pdf_path}")
    
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"   Found {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            print(f"   ✓ Page {i+1} extracted")
    
    print(f"   Total characters: {len(full_text)}")
    return full_text


# ============================================================================
# STEP 2: Chunk Text by Section Headings
# ============================================================================

def chunk_by_sections(text):
    """
    Split text into chunks based on numbered section headings.
    Pattern: "1. Title", "2. Title", etc.
    """
    print("\n✂️  Chunking text by section headings...")
    
    # Pattern to match section headings like "1. How to Change Your Password"
    # Matches: number + period + space + title text
    section_pattern = r'(\d+\.\s+[A-Z][^\n]+)'
    
    # Find all section headings and their positions
    matches = list(re.finditer(section_pattern, text))
    
    if not matches:
        print("   ⚠️  No section headings found, treating entire text as one chunk")
        return [{"title": "Full Document", "content": text, "section_num": 0}]
    
    chunks = []
    
    for i, match in enumerate(matches):
        # Get section title
        title = match.group(1).strip()
        
        # Get section content (from this heading to the next, or end of text)
        start_pos = match.start()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        content = text[start_pos:end_pos].strip()
        
        # Extract section number
        section_num = int(re.match(r'(\d+)', title).group(1))
        
        chunks.append({
            "title": title,
            "content": content,
            "section_num": section_num
        })
        
        print(f"   ✓ Section {section_num}: {title[:50]}...")
    
    print(f"\n   Total chunks created: {len(chunks)}")
    return chunks


# ============================================================================
# STEP 3: Create Embeddings with Sentence Transformers
# ============================================================================

def create_embeddings(chunks, model_name):
    """Create embeddings for each chunk using sentence-transformers"""
    print(f"\n🧠 Creating embeddings with {model_name}...")
    
    # Load the model
    print(f"   Loading model...")
    model = SentenceTransformer(model_name)
    
    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Create embeddings for each chunk's content
    texts = [chunk["content"] for chunk in chunks]
    
    print(f"   Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"   ✓ Embeddings shape: {embeddings.shape}")
    return embeddings, model


# ============================================================================
# STEP 4: Create FAISS Index
# ============================================================================

def create_faiss_index(embeddings):
    """Create FAISS index from embeddings"""
    print("\n📊 Creating FAISS index...")
    
    # Get embedding dimension
    dimension = embeddings.shape[1]
    
    # Create a flat L2 index (exact search)
    # For larger datasets, consider IndexIVFFlat for faster approximate search
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to the index
    index.add(embeddings)
    
    print(f"   ✓ Index created with {index.ntotal} vectors")
    print(f"   Index type: Flat L2 (exact search)")
    
    return index


# ============================================================================
# STEP 5: Save Index and Chunks
# ============================================================================

def save_index_and_chunks(index, chunks, index_path, chunks_path):
    """Save FAISS index and chunks to disk"""
    print("\n💾 Saving to disk...")
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"   ✓ FAISS index saved: {index_path}")
    
    # Save chunks metadata
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"   ✓ Chunks metadata saved: {chunks_path}")


# ============================================================================
# STEP 6: Test Search Function
# ============================================================================

def test_search(index, chunks, model, query, top_k=3):
    """Test the search functionality"""
    print(f"\n🔍 Test Search: '{query}'")
    print("-" * 60)
    
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"   Top {top_k} results:\n")
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks[idx]
        score = 1 / (1 + dist)  # Convert distance to similarity score
        
        print(f"   {i+1}. [{score:.3f}] {chunk['title']}")
        print(f"      Preview: {chunk['content'][:100]}...")
        print()
        
        results.append({
            "title": chunk["title"],
            "content": chunk["content"],
            "score": score,
            "section_num": chunk["section_num"]
        })
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 70)
    print("🏗️  IT Support Knowledge Base - Indexing Pipeline")
    print("=" * 70)
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"\n❌ Error: PDF not found at {PDF_PATH}")
        print("   Please place the IT_Support_Knowledge_Base.pdf in the same directory.")
        return
    
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(PDF_PATH)
    
    # Step 2: Chunk by sections
    chunks = chunk_by_sections(text)
    
    # Display chunks summary
    print("\n📋 Chunks Summary:")
    print("-" * 60)
    for chunk in chunks:
        print(f"   Section {chunk['section_num']}: {chunk['title'][:50]}...")
        print(f"      Content length: {len(chunk['content'])} chars")
    
    # Step 3: Create embeddings
    embeddings, model = create_embeddings(chunks, EMBEDDING_MODEL)
    
    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Step 5: Save everything
    save_index_and_chunks(index, chunks, INDEX_PATH, CHUNKS_PATH)
    
    # Step 6: Test with sample queries
    print("\n" + "=" * 70)
    print("🧪 Testing Search Functionality")
    print("=" * 70)
    
    test_queries = [
        "how do I change my password",
        "my screen is frozen what should I do",
        "my account is locked",
        "how to connect to VPN from home",
        "I need to install new software"
    ]
    
    for query in test_queries:
        test_search(index, chunks, model, query, top_k=2)
        print()
    
    print("=" * 70)
    print("✅ Indexing Complete!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"   📊 {INDEX_PATH} - FAISS index")
    print(f"   📄 {CHUNKS_PATH} - Chunks metadata")
    print(f"\nYou can now use these files in your RAG application.")


# ============================================================================
# UTILITY FUNCTION: Load existing index
# ============================================================================

def load_index_and_chunks(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    """Load existing FAISS index and chunks from disk"""
    print("📂 Loading existing index...")
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    print(f"   ✓ FAISS index loaded: {index.ntotal} vectors")
    
    # Load chunks
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"   ✓ Chunks loaded: {len(chunks)} sections")
    
    # Load model for queries
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   ✓ Model loaded: {EMBEDDING_MODEL}")
    
    return index, chunks, model


def search_knowledge_base(query, index, chunks, model, top_k=3):
    """
    Search the knowledge base for relevant sections.
    Returns list of relevant chunks with scores.
    """
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        score = 1 / (1 + dist)  # Convert distance to similarity score
        
        results.append({
            "title": chunk["title"],
            "content": chunk["content"],
            "score": score,
            "section_num": chunk["section_num"]
        })
    
    return results


if __name__ == "__main__":
    main()