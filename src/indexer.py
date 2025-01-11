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
        """
        Calcola l'hash MD5 del contenuto del file e del timestamp di modifica.
        
        :param file_path: Path - Il percorso del file di cui calcolare l'hash.
        :return: str - L'hash MD5 del contenuto del file e del timestamp di modifica.
        """
        content = file_path.read_bytes()
        modified_time = str(file_path.stat().st_mtime)
        return hashlib.md5(content + modified_time.encode()).hexdigest()

    def _needs_indexing(self, file_path: str) -> bool:
        """
        Verifica se un documento necessita di essere reindicizzato.
        
        :param file_path: str - Il percorso del file da verificare.
        :return: bool - True se il documento necessita di reindicizzazione, False altrimenti.
        """
        path = Path(file_path)
        if not path.exists():
            return False
        
        current_hash = self._calculate_hash(path)
        
        # Cerca il documento in Chroma
        try:
            result = self.collection.get(
                ids=[file_path],
                include=['metadatas']
            )
            print(f"results: {results}")
            
            if result and result['metadatas']:
                stored_hash = result['metadatas'][0].get('content_hash')
                return stored_hash != current_hash
                
            return True  # Il documento non esiste ancora in Chroma
            
        except Exception:
            return True  # In caso di errore, meglio reindicizzare

    def search_by_path(self, file_path: str) -> dict | None:
        """
        Cerca un documento in Chroma utilizzando il percorso del file.
        
        :param file_path: str - Il percorso del file da cercare.
        :return: dict | None - I risultati della ricerca o None se non trovato.
        """
        try:
            result = self.collection.get(
                ids=[file_path],
                include=['documents', 'metadatas']
            )
            
            # Verifica se sono stati trovati risultati
            if not result['ids']:  # Se non ci sono ID nei risultati
                print(f"Nessun documento trovato per il percorso: {file_path}")
                return None

            return result
            
        except Exception as e:
            print(f"Errore durante la ricerca del documento: {str(e)}")
            return None

    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Estrae il testo da un file DOCX in modo robusto, gestendo eventuali errori.
        
        :param file_path: str - Il percorso del file DOCX da cui estrarre il testo.
        :return: str - Il testo estratto dal file DOCX.
        """
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Errore nell'elaborazione del file {file_path}: {str(e)}")
            return ""

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide i documenti in chunks più piccoli.
        
        :param documents: List[Document] - La lista dei documenti da dividere in chunks.
        :return: List[Document] - La lista dei documenti divisi in chunks.
        """
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Estratti {len(chunks)} chunks da {len(documents)} documenti")
        return chunks

    def _process_documents(self, folder_path: str) -> List[Document]:
        """
        Processa tutti i documenti DOCX in una cartella e le sue sottocartelle.
        
        :param folder_path: str - Il percorso della cartella contenente i documenti DOCX.
        :return: List[Document] - La lista dei documenti processati.
        """
        documents = []
        for dir_path in glob.glob(os.path.join(folder_path, '**/'), recursive=True):
            for file_path in glob.glob(os.path.join(dir_path, '*.docx')):
                if self._needs_indexing(file_path=file_path):
                    text = self._extract_text_from_docx(file_path)
                    if text:
                        document = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "date_processed": str(datetime.now()),
                                "content_hash": self._calculate_hash(Path(file_path))
                            }
                        )
                        documents.append(document)
                else:
                    print(f"skipped {file_path}")

        # Split documents in chunks
        #chunks = self._chunk_documents(documents)

        return documents
    
    def clean_up(self):
        """Pulisce il vectorstore esistente."""
        self.vectorstore = Chroma.from_documents(documents=[], persist_directory=self.persist_directory)

    def index_knowledge_base(self, folder_path: str):
        """
        Indicizza i documenti in Chroma.
        
        :param folder_path: str - Il percorso della cartella contenente i documenti da indicizzare.
        """

        # Processa i documenti
        documents = self._process_documents(folder_path)
        ids = [doc.metadata["source"] for doc in documents]

        print(f"Processati {len(documents)} documenti/chunk")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(
            ids=ids,
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Vectorstore creato con {self.vectorstore._collection.count()} documents")
        
        self.collection = self.vectorstore._collection
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        print(f"Il vectorstore ha {dimensions:,} dimensioni")