# rag/evaluation.py
"""
Sistema de evaluación del pipeline RAG.
Incluye métricas de retrieval (Recall@k, MRR, MAP, NDCG)
y métricas de generación end-to-end (BLEU, ROUGE-L, METEOR, Citation Rate,
Ignorance Rate).
 
DATASET
-------
El dataset por defecto está en EVAL_SET (al final de este módulo).
Para ampliar o sustituirlo sin tocar el código, crea un fichero JSON
con la siguiente estructura y pásalo a las funciones con el parámetro
`eval_set`:
 
    [
      {
        "q": "¿Qué es la administración?",
        "keywords": ["administración", "planificación"],
        "answer_keywords": ["función", "organizar"],
        "reference_answer": "La administración es el proceso de planificar...",
        "is_negative": false
      },
      ...
    ]
 
`reference_answer` es opcional pero necesario para que BLEU/ROUGE/METEOR
sean significativos. Las queries negativas no necesitan `reference_answer`.
"""
 
from __future__ import annotations
 
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
 
from langchain_core.documents import Document
#from functools import partial
 
# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------
 
@dataclass
class EvalQuery:
    q: str
    keywords: list[str]
    answer_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""        # para BLEU / ROUGE / METEOR
    is_negative: bool = False
 
 
def load_eval_set(path: str | Path) -> list[EvalQuery]:
    """Carga un dataset externo en JSON y lo convierte a EvalQuery."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        EvalQuery(
            q=item["q"],
            keywords=item.get("keywords", []),
            answer_keywords=item.get("answer_keywords", []),
            reference_answer=item.get("reference_answer", ""),
            is_negative=item.get("is_negative", False),
        )
        for item in raw
    ]
 
 
# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
 
def _tokenize(text: str) -> list[str]:
    """Tokenización simple multilenguaje (minúsculas + split sobre no-alfa)."""
    return re.findall(r"[a-záéíóúüñ]+", (text or "").lower())
 
 
def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
 
 
# ---------------------------------------------------------------------------
# Métricas de retrieval
# ---------------------------------------------------------------------------
 
def _is_hit(doc: Document, ev: EvalQuery) -> bool:
    """Comprueba si un documento contiene alguna keyword de la query."""
    t = (doc.page_content or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in ev.keywords)
 
 
def _average_precision(docs: list[Document], ev: EvalQuery) -> float:
    """Average Precision para una query (usada en MAP)."""
    hits = 0
    precision_sum = 0.0
    for i, d in enumerate(docs, start=1):
        if _is_hit(d, ev):
            hits += 1
            precision_sum += hits / i
    # Normalizamos por min(k, n_relevantes_posibles)
    # Dado que no conocemos el total de relevantes en el corpus,
    # normalizamos por el número de hits encontrados (AP clásico truncado).
    return precision_sum / hits if hits else 0.0
 
 
def _dcg(docs: list[Document], ev: EvalQuery) -> float:
    return sum(
        _is_hit(d, ev) / math.log2(i + 1)
        for i, d in enumerate(docs, start=2)   # i empieza en 2 → log2(2)=1
    )
 
 
def _ndcg(docs: list[Document], ev: EvalQuery) -> float:
    """NDCG@k con relevancia binaria."""
    actual = _dcg(docs, ev)
    # IDCG: todos los documentos relevantes al principio
    n_relevant = sum(1 for d in docs if _is_hit(d, ev))
    ideal = sum(1 / math.log2(i + 2) for i in range(n_relevant))
    return actual / ideal if ideal else 0.0
 
# Métricas de generación
 
def _answer_relevant(response: str, ev: EvalQuery) -> bool:
    r = (response or "").lower()
    return bool(ev.answer_keywords) and any(
        re.search(rf"\b{re.escape(k)}\b", r) for k in ev.answer_keywords
    )
 
 
def _has_valid_citations(response: str) -> bool:
    return bool(re.findall(r"\[doc\s*\d+\]", response or "", flags=re.I))
 
 
def _admits_ignorance(response: str) -> bool:
    markers = [
        "no dispongo", "no tengo", "no hay información",
        "no se menciona", "no aparece", "no encuentro",
        "evidencia suficiente",
    ]
    r = (response or "").lower()
    return any(m in r for m in markers)
 

def _clean_for_metrics(text: str) -> str:
    """
    Elimina ruido estructural del RAG antes de calcular métricas léxicas.
    Conserva solo el contenido informativo de la respuesta.
    """
    # Eliminar bloque de fuentes (--- \nFuentes...)
    text = re.split(r'\n---\n', text)[0]
    # Eliminar citas [DOC n]
    text = re.sub(r'\[DOC\s*\d+\]', '', text, flags=re.I)
    # Eliminar bloque Key Strategic Insight con su cita entrecomillada
    text = re.sub(r'Key Strategic Insight\s*["\u201c][^"\u201d]*["\u201d]', '', text, flags=re.S)
    # Eliminar asteriscos de markdown
    text = re.sub(r'\*+', '', text)
    # Colapsar espacios múltiples
    return re.sub(r'\s+', ' ', text).strip()

# BLEU
 
def bleu(hypothesis: str, reference: str, max_n: int = 2) -> float:
    """
    BLEU-2
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
 
    if not hyp_tokens or not ref_tokens:
        return 0.0
 
    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not hyp_ng:
            return 0.0
        ref_counts = Counter(ref_ng)
        clipped = sum(min(cnt, ref_counts[ng]) for ng, cnt in Counter(hyp_ng).items())
        precision = clipped / len(hyp_ng)
        if precision == 0:
            return 0.0
        log_sum += (1 / max_n) * math.log(precision)
 
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    return bp * math.exp(log_sum)
 
 
# ROUGE-L
 
def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L basado en longest common subsequence (LCS).
    """
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
 
    if not hyp or not ref:
        return 0.0
 
    # LCS length con DP
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
 
    r = lcs / m
    p = lcs / n
    return (2 * p * r) / (p + r) if (p + r) else 0.0
 
 
# METEOR
 
def meteor(hypothesis: str, reference: str, alpha: float = 0.9, gamma: float = 0.5, delta: float = 0.5) -> float:
    """
    METEOR simplificado (sólo coincidencia exacta de stems, sin WordNet).
    """
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
 
    if not hyp or not ref:
        return 0.0
 
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    matches = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts if w in ref_counts)
 
    if matches == 0:
        return 0.0
 
    p = matches / len(hyp)
    r = matches / len(ref)
    fm = (p * r) / (alpha * p + (1 - alpha) * r)
 
    # Penalización por fragmentación (aproximación: chunks consecutivos en hyp)
    ref_set = set(ref)
    in_chunk = False
    chunks = 0
    for w in hyp:
        if w in ref_set:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
 
    pen = gamma * (chunks / matches) ** delta
    return fm * (1 - pen)

# Similitud coseno con embeddings

def semantic_similarity(hypothesis: str, reference: str, embeddings_model) -> float:
    """
    Similitud coseno entre embeddings de hipótesis y referencia.
    Reutiliza el modelo nomic-embed-text ya cargado en el pipeline,
    sin añadir dependencias adicionales.

    Ventaja sobre BLEU/ROUGE: insensible a diferencias de longitud y
    vocabulario, captura similitud conceptual real.
    """
    import numpy as np
    h = np.array(embeddings_model.embed_query(hypothesis))
    r = np.array(embeddings_model.embed_query(reference))
    norm_h = np.linalg.norm(h)
    norm_r = np.linalg.norm(r)
    if norm_h == 0 or norm_r == 0:
        return 0.0
    return float(np.dot(h, r) / (norm_h * norm_r))
 
 
# ---------------------------------------------------------------------------
# Evaluación de retrieval
# ---------------------------------------------------------------------------
 
def eval_retrieval(
    name: str,
    retrieve_fn: Callable,
    eval_set: list[EvalQuery] | None = None,
    k: int = 12,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evalúa el componente de recuperación.
 
    Métricas:
      - Recall@k   : fracción de queries positivas con ≥1 doc relevante en top-k
      - MRR        : Mean Reciprocal Rank
      - MAP        : Mean Average Precision
      - NDCG@k     : Normalized Discounted Cumulative Gain (relevancia binaria)
      - Specificity: fracción de queries negativas sin ningún hit (true negatives)
    """
    if eval_set is None:
        eval_set = EVAL_SET
 
    positive = [ev for ev in eval_set if not ev.is_negative]
    negative = [ev for ev in eval_set if ev.is_negative]
 
    rr_sum = ap_sum = ndcg_sum = hits = 0
    true_negatives = 0
    details_positive: list[dict] = []
    details_negative: list[dict] = []
 
    for ev in positive:
        docs = retrieve_fn(ev.q, max_docs=k)
 
        first_hit_rank: Optional[int] = None
        for i, d in enumerate(docs, start=1):
            if _is_hit(d, ev):
                first_hit_rank = i
                break
 
        hit = first_hit_rank is not None
        if hit:
            hits += 1
            rr_sum += 1.0 / first_hit_rank
 
        ap = _average_precision(docs, ev)
        ap_sum += ap
 
        ndcg = _ndcg(docs, ev)
        ndcg_sum += ndcg
 
        details_positive.append({
            "query": ev.q,
            "hit": hit,
            "rank": first_hit_rank,
            "ap": round(ap, 4),
            "ndcg": round(ndcg, 4),
            "status": f"✓ hit@{first_hit_rank}" if hit else "✗ miss",
        })
        print("DETAILS POSITIVE:", details_positive)
 
    for ev in negative:
        docs = retrieve_fn(ev.q, max_docs=k)
        any_hit = any(_is_hit(d, ev) for d in docs)
        if not any_hit:
            true_negatives += 1
        details_negative.append({
            "query": ev.q,
            "correct": not any_hit,
            "status": "✓ correcto" if not any_hit else "✗ falso positivo",
        })

        print("DETAILS NEGATIVE:", details_negative)
 
    n_pos = max(len(positive), 1)
    n_neg = max(len(negative), 1)
 
    return {
        "name": name,
        "k": k,
        # métricas principales
        "recall":      round(hits / n_pos, 4),
        "mrr":         round(rr_sum / n_pos, 4),
        "map":         round(ap_sum / n_pos, 4),
        "ndcg":        round(ndcg_sum / n_pos, 4),
        # conteos
        "hits": hits,
        "total_positive": len(positive),
        "true_negatives": true_negatives,
        "total_negative": len(negative),
        # detalles (solo si verbose)
        "details_positive": details_positive if verbose else [],
        "details_negative": details_negative if verbose else [],
    }
 
# Evaluación end-to-end
 
def eval_full(ask_fn: Callable, eval_set: list[EvalQuery] | None = None, verbose: bool = True, embeddings_model=None) -> dict[str, Any]:
    """
    Evalúa el pipeline completo (retrieval + generación).
    """
    if eval_set is None:
        eval_set = EVAL_SET
 
    positive = [ev for ev in eval_set if not ev.is_negative]
    negative = [ev for ev in eval_set if ev.is_negative]
 
    answer_hits = citation_hits = ignorance_hits = 0
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []
    meteor_scores: list[float] = []
    sem_scores:    list[float] = []
    details_positive: list[dict] = []
    details_negative: list[dict] = []
 
    for ev in positive:
        response = ask_fn(ev.q)
        response_text = response.get("answer", "") if isinstance(response, dict) else str(response)
 
        relevant  = _answer_relevant(response_text, ev)
        has_cites = _has_valid_citations(response_text)
 
        if relevant:
            answer_hits += 1
        if has_cites:
            citation_hits += 1
 
        # Métricas de texto vs. respuesta de referencia
        b = r = m = None
        if ev.reference_answer:

            # Limpiar antes de calcular métricas léxicas
            clean_text = _clean_for_metrics(response_text)

            b = bleu(response_text, ev.reference_answer)
            r = rouge_l(response_text, ev.reference_answer)
            m = meteor(response_text, ev.reference_answer)
            bleu_scores.append(b)
            rouge_scores.append(r)
            meteor_scores.append(m)

            # Similitud semántica (solo si se pasa el modelo)
            if embeddings_model is not None:
                s = semantic_similarity(clean_text, ev.reference_answer, embeddings_model)
                sem_scores.append(s)
 
        entry: dict = {
            "query":           ev.q,
            "relevant":        relevant,
            "has_citations":   has_cites,
            "bleu":            round(b, 4) if b is not None else None,
            "rouge_l":         round(r, 4) if r is not None else None,
            "meteor":          round(m, 4) if m is not None else None,
            "semantic_sim":  round(s, 4) if s is not None else None,
        }
        if verbose and (not relevant or not has_cites):
            entry["response_preview"] = response_text[:300]
        details_positive.append(entry)

    print("DETAILS POSITIVE:", details_positive)
 
    for ev in negative:
        response = ask_fn(ev.q)
        response_text = response.get("answer", "") if isinstance(response, dict) else str(response)
        admits = _admits_ignorance(response_text)
        if admits:
            ignorance_hits += 1
        entry = {"query": ev.q, "admits_ignorance": admits,
                 "status": "✓ correcto" if admits else "✗ alucinó"}
        if verbose and not admits:
            entry["response_preview"] = response_text[:300]
        details_negative.append(entry)

    print("DETAILS NEGATIVE:", details_negative)
 
    n_pos = max(len(positive), 1)
    n_neg = max(len(negative), 1)
 
    def _avg(lst: list[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 4) if lst else None
 
    return {
        # métricas de relevancia y citas
        "answer_relevancy": round(answer_hits / n_pos, 4),
        "citation_rate":    round(citation_hits / n_pos, 4),
        # métricas de texto (None si ninguna query tiene reference_answer)
        "bleu":    _avg(bleu_scores),
        "rouge_l": _avg(rouge_scores),
        "meteor":  _avg(meteor_scores),
        "semantic_sim":     _avg(sem_scores),
        # métrica negativa
        "ignorance_rate": round(ignorance_hits / n_neg, 4),
        # conteos
        "answer_hits":    answer_hits,
        "citation_hits":  citation_hits,
        "ignorance_hits": ignorance_hits,
        "total_positive": len(positive),
        "total_negative": len(negative),
        "n_with_reference": len(bleu_scores),
        # detalles
        "details_positive": details_positive if verbose else [],
        "details_negative": details_negative if verbose else [],
    }
 
 
# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
 
def run_full_eval(
    retrieval_configs: list[dict[str, Any]],
    ask_fn: Callable,
    eval_set: list[EvalQuery] | None = None,
    verbose: bool = False,
    embeddings_model = None
) -> dict[str, Any]:
    """Ejecuta evaluación completa de retrieval + end-to-end."""
    if eval_set is None:
        eval_set = EVAL_SET
 
    retrieval_results = [
        eval_retrieval(
            name=cfg["name"],
            retrieve_fn=cfg["retrieve_fn"],
            eval_set=eval_set,
            k=cfg.get("k", 12),
            verbose=True,
        )
        for cfg in retrieval_configs
    ]

    print("RETRIEVAL RESULT:", retrieval_results)

    #ask_fn = partial(ask, llm=llm, bm25=bm25, vector_retriever=vector_retriever, reranker=reranker, mq_chain=mq_chain,)
 
    e2e_result = eval_full(ask_fn=ask_fn, eval_set=eval_set, verbose=True, embeddings_model=embeddings_model)

    print("END-TO-END RESULT", e2e_result)
 
    return {
        "retrieval": retrieval_results,
        "end_to_end": e2e_result,
        "eval_set_size": {
            "positive": sum(1 for ev in eval_set if not ev.is_negative),
            "negative": sum(1 for ev in eval_set if ev.is_negative),
        },
    }

_DEFAULT_EVAL_PATH = Path(__file__).parent / "eval_data" / "default_eval_set.json"

def _load_default_eval_set() -> list[EvalQuery]:
    if _DEFAULT_EVAL_PATH.exists():
        return load_eval_set(_DEFAULT_EVAL_PATH)
    raise FileNotFoundError(
        f"No se encontró el dataset de evaluación en {_DEFAULT_EVAL_PATH}. "
        "Ejecútalo una vez con scripts/export_eval_set.py para generarlo."
    )

EVAL_SET: list[EvalQuery] = _load_default_eval_set()