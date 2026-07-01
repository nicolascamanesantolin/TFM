import re
import streamlit as st
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from .retriever import retrieve_hybrid_multiquery, dedupe_docs, reciprocal_rank_fusion
from .reranker  import rerank as _rerank

# Función que carga y cachea el modelo LLM de Ollama
@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3.1")

# Prompt estricto y principal para la generación de la respuesta final a partir de la pregunta del usuario del RAG
prompt_strict = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente experto en gestión empresarial. "
     "Responde SIEMPRE en el mismo idioma que la pregunta del usuario. "
     "Responde ÚNICAMENTE usando el CONTEXTO proporcionado. "
     "Cada frase informativa debe incluir al menos una cita explícita [DOC n]. "
     "Centra tu respuesta en el conocimiento de los fragmentos [DOC n], sin añadir conocimiento externo. "
     "Prohibido contradecir las afirmaciones presentes en los fragmentos [DOC n]. "
     "Si el contexto no permite responder con seguridad, escribe exactamente: "
     "No dispongo de evidencia suficiente en los documentos indexados. "
     "NO escribas sección 'Fuentes' ni listas bibliográficas al final de tu respuesta."),
    ("human", "Pregunta: {question}\n\nCONTEXTO:\n{context}"),
])

# Función que formatea una lista de documentos en texto estructurado con fuente, página y contenido
def format_docs_strict(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        md  = d.metadata or {}
        src = md.get("original_filename") or Path(str(md.get("source", "unknown"))).name
        page = md.get("page", "?")
        lines.append(f"[DOC {i}] Fuente: {src} | Página: {page}\n{d.page_content.strip()}")
    return "\n\n".join(lines)

# Validador de la existencia de citas en el texto pasado como parámetro
def _citations_valid(text, n_docs):
    refs = re.findall(r"\[DOC\s*(\d+)\]", text or "", flags=re.I)
    return bool(refs) and all(1 <= int(r) <= n_docs for r in refs)

# Función que extrae todos los índices numéricos de las referencias presentes en el texto pasado como parámetro
def _extract_doc_indices(text):
    return {int(x) for x in re.findall(r"\[DOC\s*(\d+)\]", text or "", flags=re.I)}

# Función que genera un bloque de texto con las fuentes a partir del parámetro de documentos que recibe
def format_sources_block(docs, used_indices=None):
    n = len(docs)
    order = sorted(i for i in used_indices if 1 <= i <= n) if used_indices else list(range(1, n+1))
    lines = ["---", "Fuentes (metadatos del índice):"]
    for i in order:
        md  = docs[i-1].metadata or {}
        src = md.get("original_filename") or Path(str(md.get("source", "unknown"))).name
        lines.append(f"- [DOC {i}] {src} | Página {md.get('page', '?')}")
    return "\n".join(lines)


_INSUFFICIENT = "No dispongo de evidencia suficiente en los documentos indexados."

def _generate_answer(question: str, docs: list, llm, prompt) -> tuple[str, set]:
    """Llama al LLM y valida citas. Devuelve (body, used_indices)."""
    chain = ({"context": lambda _: format_docs_strict(docs), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    body = chain.invoke(question).strip()

    if not _citations_valid(body, len(docs)):
        return (_INSUFFICIENT + " (la respuesta del modelo no incluyó citas [DOC n] válidas).",set())
    return body, _extract_doc_indices(body)

# Función principal que se encarga de orquestar el funcionamiento del RAG: recupera, rerankea y genera la respuesta con el modelo LLM
def ask(question, llm, bm25, vector_retriever, reranker, mq_chain,
        k_retrieve=15, k_rerank=8, n_queries=4, min_rerank_norm=0.22):

    docs = retrieve_hybrid_multiquery(question, bm25, vector_retriever, mq_chain, max_docs=k_retrieve, n_queries=n_queries)
    docs = _rerank(question, docs, reranker, top_n=k_rerank, min_norm_score=min_rerank_norm)

    if not docs:
        return _INSUFFICIENT

    body, used = _generate_answer(question, docs, llm, prompt_strict)
    return body + "\n\n" + format_sources_block(docs, used_indices=used or None)

def ask_with_session_docs(question, llm,
                          # Retrievers globales (base de conocimiento principal)
                          bm25_global, vector_retriever_global,
                          # Sesión del usuario (SessionDocumentProcessor)
                          session_processor, reranker, mq_chain, k_retrieve=15,
                          k_rerank=8, n_queries=4, min_rerank_norm=0.22,
                        ):
    # Recuperación global
    docs_global = retrieve_hybrid_multiquery(question, bm25_global, vector_retriever_global, mq_chain,max_docs=k_retrieve, n_queries=n_queries)

    # Recuperación de sesión (si hay documentos cargados)
    print("Recuperación sobre los docs de la sesión")
    print("session_processor:", session_processor)
    print("session_processor.bm25:", session_processor.bm25)
    docs_session = []
    if (session_processor is not None) and (session_processor.bm25 is not None):
        vector_retriever_session = session_processor.vectordb.as_retriever(search_kwargs={"k": k_retrieve})
        docs_session = retrieve_hybrid_multiquery(question, session_processor.bm25, vector_retriever_session, mq_chain, max_docs=k_retrieve, n_queries=n_queries,)
    print("Documentos de la sesión recuperados: ", docs_session)

    # Fusión con RRF ponderando más los docs de sesión
    if docs_session:
        fused = reciprocal_rank_fusion([docs_global, docs_session], weights=[0.4, 0.6])
        fused = dedupe_docs(fused, max_docs=k_retrieve * 2)
    else:
        fused = docs_global

    # Reranking sobre el pool combinado
    docs = _rerank(question, fused, reranker, top_n=k_rerank, min_norm_score=min_rerank_norm)

    if not docs:
        return _INSUFFICIENT

    body, used = _generate_answer(question, docs, llm, prompt_strict)
    used_docs = docs if not used else [ docs[i] for i in used if i < len(docs)]
    session_docs_used = sum(1 for doc in used_docs if "session_id" in doc.metadata)

    return {"answer": body, "sources": used_docs, "session_docs_used": session_docs_used, "total_docs": len(used_docs)}