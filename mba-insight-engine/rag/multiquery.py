import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Prompt para el modelo llm encargado de generar múltiples formulaciones de la pregunta dada
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente para recuperación de información. "
     "Debes producir consultas de búsqueda a partir del input que recibas. Trata de usar sinónimos. "
     "Devuelve SOLO JSON: array de strings; o {{\"consultas\":[...]}}. Sin markdown, sin explicación."),
    ("human", "Pregunta: {question}\n\nDevuelve SOLO el JSON (sin texto adicional)."),
])

# Función que crea la cadena de ejecución
def build_multi_query_chain(llm):
    return multi_query_prompt | llm | StrOutputParser()

# Función para procesar las respuestas de la generación de multiqueries
def _parse_json_list(text: str) -> list[str]:
    text = (text or "").strip()

    def _from_parsed(data) -> list[str] | None:
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for key in ("consultas", "queries", "preguntas", "items", "q"):
                v = data.get(key)
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
        return None

    try:
        data = json.loads(text)
        got = _from_parsed(data)
        if got:
            return got
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            got = _from_parsed(data)
            if got:
                return got
        except Exception:
            pass

    o0 = text.find("{")
    o1 = text.rfind("}")
    if o0 != -1 and o1 != -1 and o1 > o0:
        try:
            data = json.loads(text[o0 : o1 + 1])
            got = _from_parsed(data)
            if got:
                return got
        except Exception:
            pass

    out = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if len(t) > 3 and t[0].isdigit():
            for sep in [".", ")", ":", "-", "/"]:
                idx = t.find(sep)
                if 0 <= idx <= 3:
                    t = t[idx + 1 :].strip()
                    break
        out.append(t)
    return [x for x in out if x][:10]

# Función principal que se permite generar multiples queries a partir de una dada como parámetro
def generate_multi_queries(question: str, chain, n_queries: int = 4) -> list[str]:
    """Genera 3-5 consultas de búsqueda a partir de la pregunta."""
    n_queries = max(3, min(int(n_queries), 5))
    query_block = f"Genera exactamente {n_queries} consultas en español: reformulaciones y palabras clave.\n"

    raw = chain.invoke({"question": question, "n_queries": n_queries, "bilingual_block": query_block})
    queries = _parse_json_list(raw)

    seen = set()
    out = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n_queries:
            break

    q0 = (question or "").strip()
    if q0 and q0.lower() not in seen and len(out) < n_queries:
        out.append(q0)

    return out