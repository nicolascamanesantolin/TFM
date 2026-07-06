import json
from dataclasses import dataclass, field
from rag.evaluation import EVAL_SET
from dataclasses import asdict

@dataclass
class EvalQuery:
    q: str
    keywords: list[str]
    answer_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""        # para BLEU / ROUGE / METEOR
    is_negative: bool = False

EVAL_SET: list[EvalQuery] = [
    EvalQuery(
        q="¿Qué es la administración y cuáles son sus funciones básicas?",
        keywords=["administración", "planificación", "organización", "dirección", "control"],
        answer_keywords=["función", "planificar", "organizar", "dirigir", "coordinar"],
        reference_answer=(
            "La administración es el proceso de planificar, organizar, dirigir y controlar "
            "los recursos de una organización para alcanzar sus objetivos de forma eficiente."
        ),
    ),
    EvalQuery(
        q="¿Cómo se estructura una organización empresarial?",
        keywords=["estructura organizativa", "organigrama", "departamento", "jerarquía"],
        answer_keywords=["estructura", "departamento", "niveles", "jerarquía", "divisional"],
        reference_answer=(
            "Una organización empresarial se estructura mediante organigramas que definen "
            "jerarquías, departamentos y líneas de autoridad entre los distintos niveles."
        ),
    ),
    EvalQuery(
        q="¿Cómo se toman decisiones dentro de una organización?",
        keywords=["toma de decisiones", "decisión", "alternativas", "criterio"],
        answer_keywords=["decisión", "proceso", "alternativa", "información", "racional"],
        reference_answer=(
            "El proceso de toma de decisiones implica identificar el problema, generar "
            "alternativas, evaluar cada opción con criterios definidos y seleccionar la más adecuada."
        ),
    ),
    EvalQuery(
        q="¿Cómo se realiza un análisis del entorno interno y externo?",
        keywords=["dafo", "swot", "pestel", "análisis externo", "análisis interno", "entorno"],
        answer_keywords=["entorno", "interno", "externo", "fortalezas", "amenazas", "oportunidades"],
        reference_answer=(
            "El análisis del entorno combina herramientas como DAFO y PESTEL para identificar "
            "fortalezas y debilidades internas, y oportunidades y amenazas del entorno externo."
        ),
    ),
    EvalQuery(
        q="¿Qué diferencias hay entre estrategia en el sector público y privado?",
        keywords=["sector público", "sector privado", "administración pública", "estrategia pública"],
        answer_keywords=["diferencia", "público", "privado", "objetivo", "lucro"],
        reference_answer=(
            "En el sector privado la estrategia busca maximizar el beneficio económico, "
            "mientras que en el sector público el objetivo es el bien común y la prestación "
            "de servicios a la ciudadanía sin ánimo de lucro."
        ),
    ),
    EvalQuery(
        q="¿Qué es la contabilidad financiera y cuál es su objetivo?",
        keywords=["contabilidad financiera", "balance", "cuenta de resultados"],
        answer_keywords=["contabilidad", "financiero", "balance", "resultados", "patrimonio"],
        reference_answer=(
            "La contabilidad financiera registra, clasifica y resume las transacciones económicas "
            "de una empresa para ofrecer información patrimonial y de resultados a inversores y terceros."
        ),
    ),
    EvalQuery(
        q="¿Cómo se gestiona el capital de trabajo?",
        keywords=["capital circulante", "working capital", "liquidez", "activo corriente"],
        answer_keywords=["capital", "liquidez", "circulante", "gestión", "tesorería"],
        reference_answer=(
            "La gestión del capital de trabajo equilibra el activo corriente con el pasivo corriente "
            "para garantizar la liquidez operativa de la empresa sin inmovilizar recursos en exceso."
        ),
    ),
    EvalQuery(
        q="¿Cuáles son las principales teorías de motivación en la gestión de personas?",
        keywords=["motivación", "maslow", "herzberg", "teoría x", "teoría y", "necesidades"],
        answer_keywords=["motivación", "necesidad", "incentivo", "satisfacción", "empleado"],
        reference_answer=(
            "Las teorías clásicas de motivación incluyen la pirámide de Maslow, los factores "
            "higiénicos y motivadores de Herzberg, y la Teoría X e Y de McGregor, que explican "
            "qué impulsa el comportamiento de las personas en el trabajo."
        ),
    ),
    EvalQuery(
        q="¿Qué es el liderazgo y qué estilos existen?",
        keywords=["liderazgo", "líder", "estilo de liderazgo", "autocrático", "democrático", "laissez-faire"],
        answer_keywords=["liderazgo", "estilo", "autoridad", "equipo", "transformacional"],
        reference_answer=(
            "El liderazgo es la capacidad de influir en un grupo para alcanzar objetivos. "
            "Los principales estilos son el autocrático, el democrático y el laissez-faire, "
            "aunque también se distingue el liderazgo transformacional del transaccional."
        ),
    ),
    EvalQuery(
        q="¿Qué es la planificación estratégica y cuáles son sus etapas?",
        keywords=["planificación estratégica", "misión", "visión", "objetivos", "estrategia corporativa"],
        answer_keywords=["planificación", "misión", "visión", "objetivo", "estrategia", "etapa"],
        reference_answer=(
            "La planificación estratégica es el proceso mediante el cual una organización define "
            "su misión, visión y objetivos a largo plazo, y diseña las estrategias necesarias "
            "para alcanzarlos, evaluando el entorno y los recursos disponibles."
        ),
    ),
    EvalQuery(
        q="¿En qué consiste el control de gestión y qué herramientas utiliza?",
        keywords=["control de gestión", "cuadro de mando", "kpi", "balanced scorecard", "indicadores"],
        answer_keywords=["control", "indicador", "cuadro de mando", "desviación", "corrección"],
        reference_answer=(
            "El control de gestión supervisa que los resultados se ajusten a los objetivos "
            "planificados. Sus herramientas principales son el Cuadro de Mando Integral "
            "(Balanced Scorecard) y los KPI, que detectan desviaciones y proponen correcciones."
        ),
    ),
    EvalQuery(
        q="¿Qué tipos de estructura organizativa existen y cuándo conviene cada una?",
        keywords=["estructura funcional", "estructura matricial", "estructura divisional", "organización"],
        answer_keywords=["funcional", "divisional", "matricial", "ventaja", "inconveniente"],
        reference_answer=(
            "Las principales estructuras organizativas son la funcional, adecuada para entornos "
            "estables; la divisional, útil cuando hay múltiples líneas de negocio; y la matricial, "
            "que combina ambas para proyectos que requieren especialización y flexibilidad."
        ),
    ),
    # ── Negativas ─────────────────────────────────────────────────────────
    EvalQuery(
        q="¿Cuál es la receta tradicional de la paella valenciana?",
        keywords=["paella", "arroz", "ingredientes", "caldo", "azafrán"],
        is_negative=True,
    ),
    EvalQuery(
        q="¿Cómo funciona un motor de combustión interna?",
        keywords=["motor", "combustión", "pistón", "cilindro", "explosión"],
        is_negative=True,
    ),
    EvalQuery(
        q="¿Cuál es la distancia entre la Tierra y la Luna?",
        keywords=["luna", "tierra", "distancia", "kilómetros", "órbita lunar"],
        is_negative=True,
    ),
    EvalQuery(
        q="¿Cuáles son los mejores destinos turísticos de Asia?",
        keywords=["turismo", "asia", "japón", "destino", "viaje"],
        is_negative=True,
    ),
    EvalQuery(
        q="¿Cómo se prepara un bizcocho de chocolate?",
        keywords=["bizcocho", "chocolate", "harina", "huevo", "repostería"],
        is_negative=True,
    ),
]

with open("rag/eval_data/default_eval_set.json", "w", encoding="utf-8") as f:
    json.dump([asdict(ev) for ev in EVAL_SET], f, ensure_ascii=False, indent=2)