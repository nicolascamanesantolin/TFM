# TFM
Sistema de búsqueda inteligente mediante técnicas RAG aplicado a documentos sobre la gestión empresarial (MBA)

## Resumen

En este Trabajo de Fin de Máster se diseña e implementa un sistema de búsqueda
inteligente basado en Retrieval Augmented Generation (RAG) aplicado a un conjun-
to de documentos de gestión empresarial (MBA) en español. El objetivo principal del
proyecto es investigar la viabilidad de crear una herramienta que permita a estudiantes
y profesionales del sector formular preguntas en lenguaje natural y obtener respuestas
fundamentadas por la base de documentos constituida durante el proyecto. Consiguien-
do reducir el esfuerzo que conlleva tener que localizar de forma manual y tradicional
la información que se requiere en cada momento.
La solución implementada en el trabajo integra un pipeline de recuperación híbrida
(léxica y vectorial), una metodología de chunking orientada a mejorar la recuperabi-
lidad de la información, una estrategia de reformulación de consultas, una fusión de
rankings mediante reciprocal rank fusion y un módulo de reranking con un modelo
cross-encoder. La generación de respuestas se realiza con un modelo de lenguaje con
control de citas, de forma que el sistema referencia explícitamente las fuentes utilizadas
y reconoce la falta de evidencia cuando el corpus no contiene información suficiente.
Además, se desarrolla una interfaz web que incluye chat conversacional, gestor de
documentos y panel de analíticas y evaluación. La evaluación del sistema muestra un
rendimiento consistente tanto en recuperación como en el proceso completo.

## Estructura

mba-insight-engine/
|-- frontend/

|   \-- index.html               # Interfaz de usuario
|-- rag/

|   |-- __init__.py   
|   |-- chains.py                # Orquestación del pipeline

|   |-- evaluation.py            # Sistema de evaluación
|   |-- multiquery.py            # Expansión de consultas alternativas

|   |-- reranker.py              # Reordenamiento con cross-encoders
|   \-- retriever.py             # Estrategias de recuperación de información

|-- chroma_mba_business_es/      # Base de datos vectorial principal
|-- session_vectordbs/           # Bases de datos vectoriales por sesión

|-- user_uploads/                # Archivos subidos por usuarios
|-- document_processor.py        # Gestión de sesiones y documentos de usuario

|-- evaluation_results.json
|-- ingestion.py                 # Pipeline de indexación de documentos base

|-- main.py                      # Servidor FastAPI y coordinador general
