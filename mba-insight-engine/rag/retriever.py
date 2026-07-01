import streamlit as st
from pathlib import Path
from typing import Iterable
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from .multiquery import generate_multi_queries, build_multi_query_chain

PERSIST_DIR = "rag/chroma_mba_business_es"
COLLECTION  = "mba_business_rag"

# Función que carga el modelo de embeddings en caché
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Función que crea y carga en cache la base de datos vectorial
@st.cache_resource
def load_vectordb(_embeddings):
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings, collection_name=COLLECTION,)

# Función que construye y carga el retriever
@st.cache_resource
def load_bm25(_vectordb):
    data = _vectordb.get()
    lc_docs = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(data["documents"], data["metadatas"])]
    r = BM25Retriever.from_documents(lc_docs)
    r.k = 15
    return r

# Función para generar una clave única dado un documento
def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('source','unknown')}::p{md.get('page','?')}::{hash((d.page_content or '').strip())}"

# Función para controrlar la llamada al retriever en función de si se necesita invoke o get_relevant_documents
def _call_retriever(r, query: str):
    if hasattr(r, "invoke"):        return r.invoke(query)
    if hasattr(r, "get_relevant_documents"): return r.get_relevant_documents(query)
    raise TypeError(f"Retriever no soportado: {type(r)}")

# Función para fusionar los rankigs de un conjunto de documentos recuperados mediante Reciprocal Rank Fusion
def reciprocal_rank_fusion(result_lists: list[list[Document]], weights: list[float] | None = None, k: int = 60,) -> list[Document]:
    """Combina rankings de múltiples retrievers mediante RRF."""
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for w, docs in zip(weights, result_lists):
        for rank, d in enumerate(docs, start=1):
            key = _doc_key(d)
            docs_by_key.setdefault(key, d)
            scores[key] = scores.get(key, 0.0) + float(w) * (1.0 / (k + rank))

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [docs_by_key[key] for key in ranked_keys]

# Función para eliminar los docuemtnos duplicados (post RRF)
def dedupe_docs(docs: Iterable[Document], max_docs: int = 12) -> list[Document]:
    seen = set()
    out = []
    for d in docs:
        k_ = _doc_key(d)
        if k_ in seen:
            continue
        seen.add(k_)
        out.append(d)
        if len(out) >= max_docs:
            break
    return out

# Función para combiar los dos retrievers con un peso de 45% (Literal) y 55% (Semántico) para crear el Retriever Híbrido
def retrieve_hybrid(question, bm25, vector_retriever, max_docs=12, weights=(0.45, 0.55)) -> list[Document]:
    #weights = (0.2, 0.8)
    bm25_docs = _call_retriever(bm25, question)
    #print("Documentos recuperados con BM25:")
    #print(bm25_docs)
    vec_docs  = _call_retriever(vector_retriever, question)
    #print("Documentos recuperados con Vector:")
    #print(vec_docs)
    fused = reciprocal_rank_fusion([bm25_docs, vec_docs], weights=list(weights))
    return dedupe_docs(fused, max_docs=max_docs)

# Genera versiones de question y ejecuta el proceso de recuperación a partir de las funciones auxiliares anteriores
def retrieve_hybrid_multiquery(question, bm25, vector_retriever, chain, max_docs=48, n_queries=4, weights=(0.45, 0.55)) -> list[Document]:
    print("Retriever multiquery")
    queries = generate_multi_queries(question, chain, n_queries=n_queries)
    print("Queries", queries)
    if question not in queries:
        queries.append(question)
    result_lists = [retrieve_hybrid(q, bm25, vector_retriever, max_docs=max(18, max_docs//2+8), weights=weights) for q in queries]
    return dedupe_docs(reciprocal_rank_fusion(result_lists), max_docs=max_docs)