# rag/evaluation.py
"""
Sistema de evaluación del pipeline RAG.
Incluye métricas de retrieval y end-to-end.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any
from langchain_core.documents import Document

# DATASET DE EVALUACIÓN POR DEFECTO
@dataclass
class EvalQuery:
    q: str
    keywords: list[str]
    answer_keywords: list[str] = field(default_factory=list)
    is_negative: bool = False


EVAL_SET: list[EvalQuery] = [
    # Queries positivas
    EvalQuery(
        q="¿Qué es la administración y cuáles son sus funciones básicas?",
        keywords=["administración", "planificación", "organización", "dirección", "control", "management"],
        answer_keywords=["función", "planificar", "organizar", "dirigir", "coordinar"],
    ),
    EvalQuery(
        q="¿Cómo se estructura una organización empresarial?",
        keywords=["estructura organizativa", "organigrama", "departamento", "jerarquía", "organización"],
        answer_keywords=["estructura", "departamento", "niveles", "jerarquía", "divisional"],
    ),
    EvalQuery(
        q="¿Cómo se toman decisiones dentro de una organización?",
        keywords=["toma de decisiones", "decision making", "decisión", "alternativas", "criterio"],
        answer_keywords=["decisión", "proceso", "alternativa", "información", "racional"],
    ),
    EvalQuery(
        q="¿Cómo se realiza un análisis del entorno (interno y externo)?",
        keywords=["dafo", "swot", "pestel", "análisis externo", "análisis interno", "entorno"],
        answer_keywords=["entorno", "interno", "externo", "fortalezas", "amenazas", "oportunidades"],
    ),
    EvalQuery(
        q="¿Qué diferencias hay entre estrategia en el sector público y privado?",
        keywords=["sector público", "sector privado", "administración pública", "estrategia pública"],
        answer_keywords=["diferencia", "público", "privado", "objetivo", "lucro"],
    ),
    EvalQuery(
        q="¿Qué es la contabilidad financiera y cuál es su objetivo?",
        keywords=["contabilidad financiera", "financial accounting", "balance", "cuenta de resultados"],
        answer_keywords=["contabilidad", "financiero", "balance", "resultados", "patrimonio"],
    ),
    EvalQuery(
        q="¿Cómo se gestiona el capital de trabajo?",
        keywords=["capital circulante", "working capital", "liquidez", "activo corriente"],
        answer_keywords=["capital", "liquidez", "circulante", "gestión", "tesorería"],
    ),
    # Queries negativas
    EvalQuery(
        q="¿Cuál es la receta tradicional de la paella valenciana?",
        keywords=["paella", "arroz", "valenciana", "ingredientes"],
        answer_keywords=[],
        is_negative=True,
    ),
    EvalQuery(
        q="¿Cómo funciona un motor de combustión interna?",
        keywords=["motor", "combustión", "pistón", "cilindro"],
        answer_keywords=[],
        is_negative=True,
    ),
]

def _is_hit(d: Document, ev: EvalQuery) -> bool:
    """Comprueba si un documento contiene alguna keyword."""
    t = (d.page_content or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in ev.keywords)

def _answer_relevant(response: str, ev: EvalQuery) -> bool:
    """Comprueba si la respuesta contiene keywords esperadas."""
    if not ev.answer_keywords:
        return False
    r = (response or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", r) for k in ev.answer_keywords)

def _has_valid_citations(response: str) -> bool:
    """Comprueba si la respuesta incluye citas [DOC n]."""
    return bool(re.findall(r"\[doc\s*\d+\]", response or "", flags=re.I))

def _admits_ignorance(response: str) -> bool:
    """Comprueba si el modelo reconoce que no tiene información."""
    markers = ["no dispongo", "no tengo", "no hay información", "no se menciona", "no aparece"]
    r = (response or "").lower()
    return any(m in r for m in markers)

def eval_retrieval(name: str, retrieve_fn: Callable, eval_set: List[EvalQuery] = None, k: int = 12, verbose: bool = False) -> Dict[str, Any]:
    """Evalúa métricas de retrieval."""
    if eval_set is None:
        eval_set = EVAL_SET
    
    positive = [ev for ev in eval_set if not ev.is_negative]
    negative = [ev for ev in eval_set if ev.is_negative]

    hits = 0
    rr_sum = 0.0
    true_negatives = 0
    details_positive = []
    details_negative = []

    for ev in positive:
        docs = retrieve_fn(ev.q, max_docs=k)
        first_hit_rank = None
        for i, d in enumerate(docs, start=1):
            if _is_hit(d, ev):
                first_hit_rank = i
                break

        hit = first_hit_rank is not None
        if hit:
            hits += 1
            rr_sum += 1.0 / first_hit_rank

        details_positive.append({
            "query": ev.q,
            "hit": hit,
            "rank": first_hit_rank,
            "status": f"✓ hit@{first_hit_rank}" if hit else "✗ miss"
        })

    for ev in negative:
        docs = retrieve_fn(ev.q, max_docs=k)
        any_hit = any(_is_hit(d, ev) for d in docs)
        if not any_hit:
            true_negatives += 1

        details_negative.append({
            "query": ev.q,
            "correct": not any_hit,
            "status": "✓ correcto" if not any_hit else "✗ falso positivo"
        })

    n_pos = max(len(positive), 1)
    n_neg = max(len(negative), 1)

    return {"name": name, "k": k, "recall": hits / n_pos, "mrr": rr_sum / n_pos, "specificity": true_negatives / n_neg,
            "hits": hits, "total_positive": len(positive), "true_negatives": true_negatives, "total_negative": len(negative),
            "details_positive": details_positive if verbose else [], "details_negative": details_negative if verbose else []
            }


def eval_full(ask_fn: Callable,eval_set: List[EvalQuery] = None,verbose: bool = True) -> Dict[str, Any]:
    """Evalúa métricas end-to-end."""
    if eval_set is None:
        eval_set = EVAL_SET
    
    positive = [ev for ev in eval_set if not ev.is_negative]
    negative = [ev for ev in eval_set if ev.is_negative]

    answer_hits = 0
    citation_hits = 0
    ignorance_hits = 0
    details_positive = []
    details_negative = []

    for ev in positive:
        response = ask_fn(ev.q)
        
        if isinstance(response, dict):
            response_text = response.get("answer", "")
        else:
            response_text = str(response)

        relevant = _answer_relevant(response_text, ev)
        has_cites = _has_valid_citations(response_text)

        if relevant:
            answer_hits += 1
        if has_cites:
            citation_hits += 1

        details_positive.append({
            "query": ev.q,
            "relevant": relevant,
            "has_citations": has_cites,
            "response_preview": response_text[:200] if verbose and (not relevant or not has_cites) else None
        })

    for ev in negative:
        response = ask_fn(ev.q)
        
        if isinstance(response, dict):
            response_text = response.get("answer", "")
        else:
            response_text = str(response)
            
        admits = _admits_ignorance(response_text)

        if admits:
            ignorance_hits += 1

        details_negative.append({
            "query": ev.q,
            "admits_ignorance": admits,
            "response_preview": response_text[:200] if verbose and not admits else None
        })

    n_pos = max(len(positive), 1)
    n_neg = max(len(negative), 1)

    return {"answer_relevancy": answer_hits / n_pos, "citation_rate": citation_hits / n_pos, "ignorance_rate": ignorance_hits / n_neg,
            "answer_hits": answer_hits, "citation_hits": citation_hits, "ignorance_hits": ignorance_hits, "total_positive": len(positive), "total_negative": len(negative),
            "details_positive": details_positive if verbose else [], "details_negative": details_negative if verbose else [],}

def run_full_eval(retrieval_configs: List[Dict[str, Any]], ask_fn: Callable, eval_set: List[EvalQuery] = None, verbose: bool = False) -> Dict[str, Any]:
    """Ejecuta evaluación completa."""
    if eval_set is None:
        eval_set = EVAL_SET
    
    retrieval_results = []
    
    for config in retrieval_configs:
        result = eval_retrieval(
            name=config["name"],
            retrieve_fn=config["retrieve_fn"],
            eval_set=eval_set,
            k=config.get("k", 12),
            verbose=verbose
        )
        retrieval_results.append(result)
    
    e2e_result = eval_full(
        ask_fn=ask_fn,
        eval_set=eval_set,
        verbose=verbose
    )
    
    return {"retrieval": retrieval_results, "end_to_end": e2e_result,
            "eval_set_size": {"positive": len([ev for ev in eval_set if not ev.is_negative]),
                              "negative": len([ev for ev in eval_set if ev.is_negative])}
            }