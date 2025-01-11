from indexer import Indexer
from assistant import Assistant

# Esempio di utilizzo
if __name__ == "__main__":
    documents_folder = "knowledge-base"
    vectorstore = None

    indexer = Indexer()

    # Crea e usa l'indexer
    try:
        indexer.index_knowledge_base(documents_folder)
        print("Indicizzazione completata con successo!")

        assistant = Assistant(vectorstore=indexer.vectorstore)
        assistant.launch_interface()
        
    except Exception as e:
        print(f"Errore durante l'indicizzazione: {str(e)}")