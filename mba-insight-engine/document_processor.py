# document_processor.py
"""
Módulo para procesar documentos subidos por usuarios y almacenarlos 
en bases de datos vectoriales específicas de sesión.
"""

from pathlib import Path
from typing import List, Optional
import shutil
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class SessionDocumentProcessor:
    """Procesador de documentos para sesiones individuales de usuario."""
    
    def __init__(
        self,
        session_id: str,
        embeddings_model,
        base_uploads_dir: str = "user_uploads",
        base_vectordb_dir: str = "session_vectordbs"
    ):
        """Inicializa el procesador para una sesión específica."""
        self.session_id = session_id
        self.embeddings_model = embeddings_model
        
        # Crear directorios específicos de la sesión
        self.session_upload_dir = Path(base_uploads_dir) / session_id
        self.session_vectordb_dir = Path(base_vectordb_dir) / session_id
        
        self.session_upload_dir.mkdir(parents=True, exist_ok=True)
        self.session_vectordb_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar o cargar la base de datos vectorial de la sesión
        self.vectordb = self._init_vectordb()
        
        # Registro de documentos procesados
        self.processed_docs = self._load_processed_docs()

        if self.vectordb._collection.count() > 0:
            self.bm25 = self._build_bm25_index()
        else:
            self.bm25 = None
    
    def _init_vectordb(self) -> Chroma:
        """Inicializa o carga la base de datos vectorial de la sesión."""
        return Chroma(
            embedding_function=self.embeddings_model,
            persist_directory=str(self.session_vectordb_dir),
            collection_name=f"session_{self.session_id}"
        )
    
    def _load_processed_docs(self) -> dict:
        """Carga el registro de documentos procesados."""
        registry_file = self.session_upload_dir / "processed_registry.txt"
        processed = {}
        
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            processed[parts[0]] = {
                                'timestamp': parts[1],
                                'chunks': int(parts[2])
                            }
        return processed
    
    def _save_processed_doc(self, filename: str, chunks_count: int):
        """Registra un documento como procesado."""
        registry_file = self.session_upload_dir / "processed_registry.txt"
        timestamp = datetime.now().isoformat()
        
        self.processed_docs[filename] = {
            'timestamp': timestamp,
            'chunks': chunks_count
        }
        
        with open(registry_file, 'a', encoding='utf-8') as f:
            f.write(f"{filename}|{timestamp}|{chunks_count}\n")
    
    def _normalize(self, text: str) -> str:
        """Normaliza el texto eliminando espacios en blanco excesivos."""
        return " ".join((text or "").split())
    
    def _load_pdf(self, pdf_path: str) -> List[Document]:
        """Carga un PDF y extrae los documentos."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"PDF cargado: {len(documents)} páginas")
        return documents
    
    def _hybrid_chunking(
        self,
        documents: List[Document],
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
        breakpoint_type: str = "percentile",
        breakpoint_amount: float = 85,
        pre_split_size: int = 6000,
        min_chunk_chars: int = 120,
    ) -> List[Document]:
        """Aplica chunking híbrido: pre-split fijo -> semantic split -> hard cap."""
        # Pre-split
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=pre_split_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pre_chunks = pre_splitter.split_documents(documents)
        
        # Semantic chunking
        semantic_splitter = SemanticChunker(
            embeddings=self.embeddings_model,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount,
        )
        
        semantic_chunks = []
        for pre_chunk in pre_chunks:
            pre_text = self._normalize(pre_chunk.page_content)
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
        
        # Hard cap
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        final_chunks = []
        oversized = 0
        
        for chunk in semantic_chunks:
            chunk.page_content = self._normalize(chunk.page_content)
            if len(chunk.page_content) < min_chunk_chars:
                continue
            
            if len(chunk.page_content) > max_chunk_size:
                oversized += 1
                sub_chunks = recursive_splitter.split_documents([chunk])
                for sc in sub_chunks:
                    sc.page_content = self._normalize(sc.page_content)
                    sc.metadata = dict(chunk.metadata or {})
                final_chunks.extend(
                    [sc for sc in sub_chunks if len(sc.page_content) >= min_chunk_chars]
                )
            else:
                final_chunks.append(chunk)
        
        # Filtrado final
        before = len(final_chunks)
        final_chunks = [
            c for c in final_chunks 
            if len(self._normalize(c.page_content)) >= min_chunk_chars
        ]
        removed = before - len(final_chunks)
        
        print(f"Chunks que superaban el límite: {oversized}")
        print(f"Eliminados por ser muy cortos (< {min_chunk_chars} chars): {removed}")
        print(f"Total chunks finales: {len(final_chunks)}")
        
        return final_chunks
    
    def _build_bm25_index(self):
        """Construye índice BM25 con los docs actuales de la sesión."""
        # Extraer todos los documentos de la BD vectorial de sesión
        collection = self.vectordb._collection
        all_data = collection.get()
        
        documents = [
            Document(
                page_content=content,
                metadata=meta or {}
            )
            for content, meta in zip(all_data['documents'], all_data['metadatas'])
        ]
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 12
        
        return bm25_retriever
    
    def process_document(self, file_path: str, original_filename: Optional[str] = None) -> dict:
        """
        Procesa un documento y lo añade a la base de datos vectorial de la sesión.
        """
        try:
            file_path = Path(file_path)
            print(file_path)
            print(self.session_upload_dir)
            filename = original_filename or file_path.name
            
            # Verificar si ya fue procesado
            if filename in self.processed_docs:
                return {
                    'success': False,
                    'filename': filename,
                    'chunks_count': 0,
                    'message': 'El documento ya ha sido procesado',
                    'error': 'ALREADY_PROCESSED'
                }
            
            # Verificar que el archivo existe
            if not file_path.exists():
                return {
                    'success': False,
                    'filename': filename,
                    'chunks_count': 0,
                    'message': 'El archivo no existe',
                    'error': 'FILE_NOT_FOUND'
                }
            
            # Verificar que es un PDF
            if file_path.suffix.lower() != '.pdf':
                return {
                    'success': False,
                    'filename': filename,
                    'chunks_count': 0,
                    'message': 'Solo se soportan archivos PDF',
                    'error': 'INVALID_FORMAT'
                }
            
            print(f"\n{'='*60}")
            print(f"Procesando documento: {filename}")
            print(f"{'='*60}")
            
            # Cargar el PDF
            documents = self._load_pdf(str(file_path))
            
            # Aplicar chunking híbrido
            chunks = self._hybrid_chunking(documents)
            
            if not chunks:
                return {
                    'success': False,
                    'filename': filename,
                    'chunks_count': 0,
                    'message': 'No se pudieron generar chunks del documento',
                    'error': 'NO_CHUNKS_GENERATED'
                }
            
            # Generar IDs únicos para los chunks
            ids = [
                f"session_{self.session_id}::{filename}::p{chunk.metadata.get('page', '?')}::c{i}"
                for i, chunk in enumerate(chunks)
            ]
            
            # Añadir metadatos adicionales
            for chunk in chunks:
                chunk.metadata['session_id'] = self.session_id
                chunk.metadata['original_filename'] = filename
                chunk.metadata['upload_timestamp'] = datetime.now().isoformat()
            
            # Añadir chunks a la base de datos vectorial
            self.vectordb.add_documents(documents=chunks,ids=ids)
            self.bm25 = self._build_bm25_index()
            
            # Registrar el documento como procesado
            self._save_processed_doc(filename, len(chunks))
            
            print(f"\n✓ Documento procesado exitosamente")
            print(f"  - Chunks generados: {len(chunks)}")
            print(f"  - Total chunks en BD: {self.vectordb._collection.count()}")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_count': len(chunks),
                'message': f'Documento procesado exitosamente ({len(chunks)} fragmentos)',
                'total_chunks_in_db': self.vectordb._collection.count()
            }
            
        except Exception as e:
            print(f"Error procesando documento: {str(e)}")
            return {
                'success': False,
                'filename': filename if 'filename' in locals() else 'unknown',
                'chunks_count': 0,
                'message': f'Error al procesar el documento: {str(e)}',
                'error': 'PROCESSING_ERROR'
            }
    
    def query(
        self,
        query_text: str,
        k: int = 5,
        filter_by_doc: Optional[str] = None
    ) -> List[Document]:
        """Realiza una consulta en la base de datos vectorial de la sesión."""
        search_kwargs = {"k": k}
        
        if filter_by_doc:
            search_kwargs["filter"] = {"original_filename": filter_by_doc}
        
        retriever = self.vectordb.as_retriever(search_kwargs=search_kwargs)
        results = retriever.invoke(query_text)
        
        return results
    
    def get_processed_documents(self) -> dict:
        """Retorna la lista de documentos procesados en esta sesión."""
        return self.processed_docs.copy()
    
    def remove_document(self, filename: str) -> dict:
        """Elimina un documento de la base de datos vectorial de la sesión."""
        try:
            if filename not in self.processed_docs:
                return {
                    'success': False,
                    'message': 'El documento no está procesado',
                    'error': 'NOT_FOUND'
                }
            
            # Obtener todos los IDs que pertenecen a este documento
            collection = self.vectordb._collection
            all_data = collection.get()
            
            ids_to_delete = [
                id_ for id_ in all_data['ids']
                if f"::{filename}::" in id_
            ]
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            
            # Eliminar del registro
            del self.processed_docs[filename]
            
            # Reescribir el archivo de registro
            registry_file = self.session_upload_dir / "processed_registry.txt"
            with open(registry_file, 'w', encoding='utf-8') as f:
                for fname, info in self.processed_docs.items():
                    f.write(f"{fname}|{info['timestamp']}|{info['chunks']}\n")
            
            return {
                'success': True,
                'message': f'Documento eliminado ({len(ids_to_delete)} chunks)',
                'chunks_deleted': len(ids_to_delete)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error al eliminar documento: {str(e)}',
                'error': 'DELETE_ERROR'
            }
    
    def clear_session(self):
        """Limpia todos los datos de la sesión."""
        try:
            if self.session_vectordb_dir.exists():
                shutil.rmtree(self.session_vectordb_dir)
            
            if self.session_upload_dir.exists():
                shutil.rmtree(self.session_upload_dir)
            
            print(f"Sesión {self.session_id} limpiada exitosamente")
            
        except Exception as e:
            print(f"Error al limpiar sesión: {str(e)}")


class SessionManager:
    """Administra múltiples sesiones de procesamiento de documentos."""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str) -> SessionDocumentProcessor:
        """Obtiene una sesión existente o crea una nueva."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionDocumentProcessor(
                session_id=session_id,
                embeddings_model=self.embeddings_model
            )
        
        return self.sessions[session_id]
    
    def remove_session(self, session_id: str):
        """Elimina una sesión y limpia sus datos."""
        if session_id in self.sessions:
            self.sessions[session_id].clear_session()
            del self.sessions[session_id]