# ingestion.py  →  python ingestion.py
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ...resto de imports (PyPDFLoader, SemanticChunker, etc.)

# Pega aquí: load_pdf(), semantic_chunking(),
#             hybrid_chunking(), print_chunk_stats(), main_chunking()


# Funciones auxiliares para el chunking

def load_pdf(pdf_path: str):
    """Carga un PDF y extrae los documentos."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDF cargado: {len(documents)} páginas")
    return documents

def semantic_chunking(documents, embeddings_model):  # => No se usa (se usa hybrid chunking)
    """
    Aplica semantic chunking usando SemanticChunker de LangChain.
    
    Breakpoint types:
    - "percentile": divide cuando la diferencia de similitud supera el percentil X (default 95)
    - "standard_deviation": divide cuando supera X desviaciones estándar
    - "interquartile": usa IQR para detectar cambios semánticos
    """
    text_splitter = SemanticChunker(
        embeddings=embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    
    full_text = "\n\n".join([doc.page_content for doc in documents]) # Unir todo el texto primero para que el chunker analice semánticamente
    chunks = text_splitter.create_documents([full_text])
    
    print(f"Chunks generados: {len(chunks)}")
    # for i, chunk in enumerate(chunks[:3]):  # Preview primeros 3
    #     print(f"\n--- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
    #     print(chunk.page_content[:200] + "...")
    
    return chunks

def print_chunk_stats(chunks):
    """Muestra estadísticas de los chunks."""
    sizes = [len(c.page_content) for c in chunks]
    print(f"\\nEstadísticas de chunks:")
    print(f"   Mínimo : {min(sizes)} chars")
    print(f"   Máximo : {max(sizes)} chars")
    print(f"   Media  : {sum(sizes) // len(sizes)} chars")
    oversized = sum(1 for s in sizes if s > 512)
    print(f"   > 512  : {oversized} chunks")

def hybrid_chunking(documents,
                    embeddings_model,
                    max_chunk_size: int = 2000,
                    chunk_overlap: int = 200,
                    breakpoint_type: str = "percentile",
                    breakpoint_amount: float = 85,
                    pre_split_size: int = 6000,
                    min_chunk_chars: int = 120,):

    def _normalize(text: str) -> str:
        return " ".join((text or "").split())

    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=pre_split_size, chunk_overlap=0, separators=["\\n\\n", "\\n", ". ", " ", ""],)
    pre_chunks = pre_splitter.split_documents(documents)

    semantic_splitter = SemanticChunker(embeddings=embeddings_model,
                                        breakpoint_threshold_type=breakpoint_type,
                                        breakpoint_threshold_amount=breakpoint_amount,)

    semantic_chunks = []
    for pre_chunk in pre_chunks:
        pre_text = _normalize(pre_chunk.page_content)
        if len(pre_text) < min_chunk_chars:
            continue

        try:
            created = semantic_splitter.create_documents([pre_text])
            for c in created:
                c.metadata = dict(pre_chunk.metadata or {})
            semantic_chunks.extend(created)
        except Exception as e:
            print(f"Error en pre-chunk, usando tal cual: {e}")
            pre_chunk.page_content = pre_text
            semantic_chunks.append(pre_chunk)

    print(f"Chunks semánticos: {len(semantic_chunks)}")

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )

    final_chunks = []
    oversized = 0

    for chunk in semantic_chunks:
        chunk.page_content = _normalize(chunk.page_content)
        if len(chunk.page_content) < min_chunk_chars:
            continue

        if len(chunk.page_content) > max_chunk_size:
            oversized += 1
            sub_chunks = recursive_splitter.split_documents([chunk])
            for sc in sub_chunks:
                sc.page_content = _normalize(sc.page_content)
                sc.metadata = dict(chunk.metadata or {})
            final_chunks.extend([sc for sc in sub_chunks if len(sc.page_content) >= min_chunk_chars])
        else:
            final_chunks.append(chunk)

    before = len(final_chunks)
    final_chunks = [c for c in final_chunks if len(_normalize(c.page_content)) >= min_chunk_chars]
    removed = before - len(final_chunks)

    print(f"Chunks que superaban el límite: {oversized}")
    print(f"Eliminados por ser muy cortos (< {min_chunk_chars} chars): {removed}")
    print(f"Total chunks finales: {len(final_chunks)}")
    print_chunk_stats(final_chunks)

    return final_chunks

# Funcion principal
def main_chunking(f_path, embeddings_model):
    all_chunks = []

    folder = Path(f_path)

    for pdf_path in folder.glob("*.pdf"):
        print(f"Procesando: {pdf_path}")

        # Carga del pdf
        documents = load_pdf(str(pdf_path))

        # Hybrid Chunking (Semantic Chunking + Fixed Chunking)
        chunks = hybrid_chunking(documents, embeddings_model)
        # chunks = semantic_chunking(documents, embeddings_model)

        all_chunks.extend(chunks)

    return all_chunks

if __name__ == "__main__":
    folder_path = "MBA Documents ES/"
    persist_dir = "chroma_mba_business_es"

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )

    all_chunks = main_chunking(folder_path, embeddings)

    ids = [
        f"{Path(str(d.metadata.get('source','unknown'))).stem}::p{d.metadata.get('page','?')}::c{i}"
        for i, d in enumerate(all_chunks)
    ]

    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=persist_dir,
        collection_name="mba_business_rag",
    )
    print(f"Indexados {vectordb._collection.count()} chunks.")