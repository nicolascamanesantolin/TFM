"""
MBA Insight Engine — FastAPI backend
Serves the RAG logic via REST + the static frontend.

Run:
    uvicorn main:app --reload --port 8000

Deps (add to requirements.txt):
    fastapi uvicorn[standard] python-multipart
    (plus your existing rag/* deps)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from document_processor import SessionManager
from langchain_ollama import OllamaEmbeddings
import os, shutil, traceback, uuid
import json
from datetime import datetime
from typing import Optional
import asyncio
from rag.chains    import load_llm, ask as rag_ask, ask_with_session_docs
from rag.retriever import load_embeddings, load_vectordb, load_bm25, retrieve_hybrid, retrieve_hybrid_multiquery
from rag.reranker  import load_reranker, rerank
from rag.multiquery import build_multi_query_chain
from rag.evaluation import run_full_eval, EVAL_SET

app = FastAPI(title="MBA Insight Engine", version="1.0")

EVAL_RESULTS_FILE = "evaluation_results.json"
evaluation_running = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

session_manager = SessionManager(embeddings)

@app.on_event("startup")
def startup():
    global embeddings, vectordb, vector_retriever, bm25, llm, reranker, mq_chain
    embeddings       = load_embeddings()
    vectordb         = load_vectordb(embeddings)
    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 80}
    )
    bm25      = load_bm25(vectordb)
    llm       = load_llm()
    reranker  = load_reranker()
    mq_chain  = build_multi_query_chain(llm)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []

class ProcessDocumentRequest(BaseModel):
    session_id: str
    filename: str

class QueryRequest(BaseModel):
    session_id: str
    question: str
    use_session_docs: bool = True  # Si usar documentos de la sesión

class EvalRequest(BaseModel):
    verbose: bool = False
    custom_queries: list | None = None

# Endpoints
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    """Serve the SPA."""
    return FileResponse("frontend/index.html")

@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """Main RAG endpoint."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag_ask(body.question, llm, bm25,vector_retriever, reranker, mq_chain)

        # ask may return a plain str or a dict with {"answer":…,"sources":[…]}
        if isinstance(result, dict):
            return AskResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", [])
            )
        return AskResponse(answer=str(result))

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="RAG pipeline error.")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF to be indexed."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/documents")
def list_documents():
    """Return the list of indexed PDFs."""
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"documents": files}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(None)):
    """Sube un archivo y lo guarda temporalmente."""
    try:
        print(session_id)
        if not session_id:
            session_id = str(uuid.uuid4())
        print(session_id)
        
        # Crear directorio temporal para la sesión
        upload_dir = Path("user_uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        print(upload_dir)
        
        # Guardar el archivo
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "message": "Archivo subido correctamente"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-document")
async def process_document(request: ProcessDocumentRequest):
    """Procesa un documento y lo añade a la BD vectorial de la sesión."""
    try:
        # Obtener o crear procesador de sesión
        processor = session_manager.get_or_create_session(request.session_id)
        
        # Procesar el documento
        file_path = Path("user_uploads") / request.session_id / request.filename
        print(file_path)
        result = processor.process_document(
            file_path=str(file_path),
            original_filename=request.filename
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/documents")
async def get_session_documents(session_id: str):
    """Obtiene la lista de documentos procesados en una sesión."""
    try:
        processor = session_manager.get_or_create_session(session_id)
        documents = processor.get_processed_documents()
        
        return {
            "success": True,
            "session_id": session_id,
            "documents": documents
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}/document/{filename}")
async def delete_document(session_id: str, filename: str):
    """Elimina un documento de la BD vectorial de la sesión."""
    try:
        processor = session_manager.get_or_create_session(session_id)
        result = processor.remove_document(filename)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    Endpoint de chat con soporte completo para documentos de sesión.
    
    Procesa consultas usando BD principal + documentos del usuario a través del MISMO pipeline RAG unificado que /ask, asegurando consistencia en arquitectura y calidad de respuestas.
    """
    # VALIDACIÓN INICIAL
    if not request.question.strip():raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        session_processor = None
        session_docs_available = 0
        
        if request.use_session_docs and request.session_id:
            try:
                #print("SESSION ID:", request.session_id)
                session_processor = session_manager.get_or_create_session(request.session_id)
                #print("SESSION PROCESSOR", session_processor)
                
                # Contar documentos disponibles en sesión
                processed_docs = session_processor.get_processed_documents()
                print("PROCESSED DOCS: ", processed_docs)
                session_docs_available = sum(info["chunks"]for info in processed_docs.values())
                #print("SESSION DOCS:", session_docs_available)
                
            except Exception as e:
                print(f"Error cargando sesión: {e}")
                # Continuar sin documentos de sesión
                session_processor = None
        
        # 2: Ejecutar pipeline RAG unificado
        #print(f"\nIniciando pipeline RAG...")
        vector_retriever = load_vectordb().as_retriever()
        bm25 = load_bm25(vector_retriever)
        result = ask_with_session_docs(request.question, llm, bm25, vector_retriever, session_processor, reranker, mq_chain, k_retrieve=15, k_rerank=8, n_queries=4, min_rerank_norm=0.22)
        print("RESULTADO:", result)
        
        # 3: Extraer datos de resultado
        answer = result["answer"]
        print("ANSWER:", answer)
        source_docs = result["sources"]
        print("SOURCE DOCS:", source_docs)
        session_docs_used = result["session_docs_used"]
        print("SESSION DOCS USED", session_docs_used)
        total_docs_used = result["total_docs"]
        print("TOTAL DOCS USED", total_docs_used)
        
        # 4: Construir lista de fuentes para respuesta
        sources_response = []
        
        for doc in source_docs:
            is_session_doc = 'session_id' in doc.metadata
            session_id_value = doc.metadata.get('session_id') if is_session_doc else None
            
            # Determinar nombre del archivo
            if is_session_doc:
                filename = doc.metadata.get('original_filename', 'Documento de sesión')
            else:
                filename = doc.metadata.get('source', 'Desconocido')
            
            source_info = {"filename": filename, "page": doc.metadata.get('page', '?'), "is_session_doc": is_session_doc,"session_id": session_id_value}
            
            sources_response.append(source_info)
        
        # 5: Construir respuesta final
        from datetime import datetime
        
        response = {
            "success": True,
            "answer": answer,
            "sources": sources_response,
            "metadata": {
                "total_docs_used": total_docs_used,
                "session_docs_used": session_docs_used,
                "session_docs_available": session_docs_available,
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        # Capturar cualquier otro error
        import traceback
        
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")
    
@app.get("/api/evaluate/results")
async def get_evaluation_results():
    """Obtiene los resultados de la última evaluación almacenada."""
    try:
        if os.path.exists(EVAL_RESULTS_FILE):
            with open(EVAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "success": True,
                "has_results": True,
                "results": data.get("results"),
                "timestamp": data.get("timestamp"),
                "duration": data.get("duration")
            }
        else:
            return {
                "success": True,
                "has_results": False,
                "message": "No hay evaluaciones previas. Ejecuta una evaluación para ver resultados."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluate/status")
async def get_evaluation_status():
    """Retorna el estado del sistema de evaluación."""
    global evaluation_running
    
    has_results = os.path.exists(EVAL_RESULTS_FILE)
    last_run = None
    
    if has_results:
        try:
            with open(EVAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            last_run = data.get("timestamp")
        except:
            pass
    
    return {
        "success": True,
        "available": True,
        "running": evaluation_running,
        "has_results": has_results,
        "last_run": last_run,
        "eval_set_size": len(EVAL_SET),
        "configurations": 3
    }

@app.get("/api/evaluate/queries")
async def get_eval_queries():
    """Obtiene las queries de evaluación actuales."""
    try:
        queries_data = []
        for eq in EVAL_SET:
            queries_data.append({
                "q": eq.q,
                "keywords": eq.keywords,
                "answer_keywords": eq.answer_keywords,
                "is_negative": eq.is_negative
            })
        
        return {
            "success": True,
            "queries": queries_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/run")
async def run_evaluation_async(request: EvalRequest):
    """Ejecuta la evaluación completa del sistema RAG con queries personalizadas."""
    global evaluation_running

    verbose = request.verbose
    custom_queries = request.custom_queries
    
    if evaluation_running:
        raise HTTPException(
            status_code=409, 
            detail="Ya hay una evaluación en ejecución. Por favor espera."
        )
    
    evaluation_running = True
    start_time = datetime.now()
    
    try:
        # Construir eval_set desde custom_queries si se proporciona
        from functools import partial
        from rag.evaluation import EvalQuery
        
        if custom_queries:
            eval_set = [
                EvalQuery(
                    q=q.get("q", ""),
                    keywords=q.get("keywords", []),
                    answer_keywords=q.get("answer_keywords", []),
                    reference_answer=q.get("reference_answer", ""),
                    is_negative=q.get("is_negative", False)
                )
                for q in custom_queries
            ]
        else:
            eval_set = EVAL_SET
        
        # ask_fn con todos los recursos ya inicializados como globales
        ask_fn = lambda q: rag_ask(q, llm, bm25, vector_retriever, reranker, mq_chain)
        
        # Definir función wrapper para retrieve con reranking
        def retrieve_reranked(question: str, max_docs: int = 12):
            docs = retrieve_hybrid_multiquery(question, bm25, vector_retriever, mq_chain, max_docs=48)
            return rerank(question, docs, reranker, top_n=max_docs)
        
        # Configuraciones de retrieval a comparar
        retrieval_configs = [
            {
                "name": "Hybrid (BM25 + Vector)",
                "retrieve_fn": lambda q, max_docs=12: retrieve_hybrid(q, bm25, vector_retriever, max_docs=max_docs),
                "k": 12
            },
            {
                "name": "Multi-query Hybrid + RRF",
                "retrieve_fn": lambda q, max_docs=12: retrieve_hybrid_multiquery(q, bm25, vector_retriever, mq_chain, max_docs=max_docs),
                "k": 12
            },
            {
                "name": "Multi-query + RRF + Reranking",
                "retrieve_fn": retrieve_reranked,
                "k": 12
            },
        ]
        
        print("\n" + "="*60)
        print("INICIANDO EVALUACIÓN DEL SISTEMA RAG")
        print(f"Queries a evaluar: {len(eval_set)}")
        print("="*60)
        
        # Ejecutar evaluación
        results = run_full_eval(
            retrieval_configs=retrieval_configs,
            ask_fn=ask_fn,
            eval_set=eval_set,
            verbose=True,
            embeddings_model=embeddings
        )

        print("RESULTS MAIN: ", results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Guardar resultados
        eval_data = {
            "results": results,
            "timestamp": start_time.isoformat(),
            "duration": duration,
            "verbose": verbose,
            "queries_used": len(eval_set)
        }
        
        with open(EVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Evaluación completada en {duration:.1f} segundos")
        print(f"  Resultados guardados en {EVAL_RESULTS_FILE}")
        print("="*60 + "\n")
        
        evaluation_running = False
        
        return {
            "success": True,
            "results": results,
            "duration": duration,
            "timestamp": start_time.isoformat()
        }
    
    except Exception as e:
        evaluation_running = False
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Static assets
if os.path.isdir("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")