import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Función que carga en caché el modelo cross-encoder para el reranking
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Función para reordenar documetos en función de su relevancia con la pregunta pasada como parámetro usando un modelo de cross-encoder
def rerank(question: str, docs: list[Document], reranker, top_n: int = 8, min_norm_score: float = 0.0) -> list[Document]:
    if not docs:
        return []
    pairs = [(question, d.page_content) for d in docs]
    raw   = [float(x) for x in reranker.predict(pairs)]
    mn, mx = min(raw), max(raw)
    span   = (mx - mn) if (mx - mn) > 1e-9 else 1.0
    norm   = [(r - mn) / span for r in raw]
    ranked = sorted(zip(norm, docs), key=lambda x: x[0], reverse=True)

    for score, doc in ranked:
        doc.metadata['rerank_score'] = round(score, 4)

    if min_norm_score <= 0:
        return [d for _, d in ranked[:top_n]]
    passed = [d for n, d in ranked if n >= min_norm_score][:top_n]
    return passed or [d for _, d in ranked[:max(1, top_n)]]