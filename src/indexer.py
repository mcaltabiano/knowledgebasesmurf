# Standard library imports
import os
import glob
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Third-party imports
from docx import Document as DocxDocument
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class Indexer:
    def __init__(self):
        self.persist_directory = "./vector_db"
        self.vectorstore = None
        self.collection = None

    def _calculate_hash(self, file_path: Path) -> str:
        content = file_path.read_bytes()
        modified_time = str(file_path.stat().st_mtime)
        return hashlib.md5(content + modified_time.encode()).hexdigest()

    def _needs_indexing(self, file_path: str) -> bool:
        """
        Verifica se un documento necessita di essere reindicizzato.
        Controlla sia il documento originale che i suoi chunks.
        
        :param file_path: str - Il percorso del file da verificare.
        :return: bool - True se il documento necessita di reindicizzazione, False altrimenti.
        """
        path = Path(file_path)
        if not path.exists():
            return False
        
        current_hash = self._calculate_hash(path)
        
        try:
            # Cerca tutti i chunks associati al documento
            results = self.collection.get(
                where={"source": file_path},
                include=['metadatas']
            )
            
            if results and results['metadatas']:
                # Verifica il content_hash del primo chunk
                # Se è diverso, il documento è stato modificato
                stored_hash = results['metadatas'][0].get('content_hash')
                return stored_hash != current_hash
                
            return True  # Il documento non esiste ancora in Chroma
            
        except Exception as e:
            print(f"Errore durante la verifica del documento: {str(e)}")
            return True

    def _process_documents(self, folder_path: str) -> List[Document]:
        """
        Processa tutti i documenti DOCX in una cartella e le sue sottocartelle,
        dividendoli in chunks.
        
        :param folder_path: str - Il percorso della cartella contenente i documenti DOCX.
        :return: List[Document] - La lista dei chunks processati.
        """
        chunks = []
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for dir_path in glob.glob(os.path.join(folder_path, '**/'), recursive=True):
            for file_path in glob.glob(os.path.join(dir_path, '*.docx')):
                if self._needs_indexing(file_path=file_path):
                    text = self._extract_text_from_docx(file_path)
                    if text:
                        content_hash = self._calculate_hash(Path(file_path))
                        # Crea un documento per il testo completo
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "date_processed": str(datetime.now()),
                                "content_hash": content_hash,
                                "chunk_index": -1  # -1 indica il documento completo
                            }
                        )
                        
                        # Dividi il documento in chunks
                        doc_chunks = text_splitter.split_documents([doc])
                        
                        # Aggiorna i metadata per ogni chunk
                        for i, chunk in enumerate(doc_chunks):
                            chunk.metadata.update({
                                "chunk_index": i,
                                "total_chunks": len(doc_chunks),
                                "content_hash": content_hash  # Mantieni lo stesso hash per tutti i chunks
                            })
                        
                        chunks.extend(doc_chunks)
                else:
                    print(f"Il documento {file_path} non necessita di reindicizzazione")

        return chunks

    def index_knowledge_base(self, folder_path: str):
        """
        Indicizza i documenti in Chroma.
        
        :param folder_path: str - Il percorso della cartella contenente i documenti da indicizzare.
        """
        # Processa i documenti e ottieni i chunks
        chunks = self._process_documents(folder_path)
        
        # Genera ID unici per ogni chunk
        ids = [f"{doc.metadata['source']}_{doc.metadata['chunk_index']}" for doc in chunks]
        
        print(f"Processati {len(chunks)} chunks")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(
            ids=ids,
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Vectorstore creato con {self.vectorstore._collection.count()} chunks")
        
        self.collection = self.vectorstore._collection