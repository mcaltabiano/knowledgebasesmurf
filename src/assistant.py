import gradio as gr
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStore

from typing import Optional
import yaml

class Assistant:
    def __init__(
        self,
        vectorstore: VectorStore,
        model_name: str = "mistral",
        retriever_k: int = 25
    ):
        """
        Inizializza l'assistente per la chat sui documenti.
        
        Args:
            vectorstore: Il VectorStore contenente i documenti indicizzati
            model_name: Il nome del modello Ollama da utilizzare
            retriever_k: Numero di documenti da recuperare per ogni query
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.retriever_k = retriever_k
        self.conversation_chain = None
        self.interface = None
        
        self._read_config()
        self._setup_chain()
    
    def _read_config(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.custom_prompt = config.get('assistant', {}).get('custom_prompt', None)

    def _setup_chain(self):
        """Configura la catena di conversazione con LLM, retriever e memoria."""
        llm = OllamaLLM(model=self.model_name)
        
        custom_prompt = getattr(self, 'custom_prompt', None)
        params = {
            "memory_key": "chat_history",
            "return_messages": True,
            **({"prompt": custom_prompt} if custom_prompt else {})
        }

        memory = ConversationBufferMemory(**params)

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retriever_k}
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
    
    def _chat(self, message: str, history: Optional[list] = None) -> str:
        """
        Gestisce una singola interazione di chat.
        
        Args:
            message: Il messaggio dell'utente
            history: La cronologia della chat (non utilizzata direttamente)
            
        Returns:
            La risposta del modello
        """
        result = self.conversation_chain.invoke({"question": message})
        return result["answer"]
    
    def launch_interface(
        self,
        share: bool = False,
        inbrowser: bool = True,
        server_name: str = "0.0.0.0",
        server_port: int = 7860
    ):
        """
        Avvia l'interfaccia Gradio per la chat.
        
        Args:
            share: Se condividere l'interfaccia pubblicamente
            inbrowser: Se aprire automaticamente nel browser
            server_name: Nome del server
            server_port: Porta del server
        """
        self.interface = gr.ChatInterface(
            fn=self._chat,
            title="Puffo assistente",
            description="Fai domande sulla tua knowledge base",
            type="messages"
        )
        
        self.interface.launch(
            share=share,
            inbrowser=inbrowser,
            server_name=server_name,
            server_port=server_port
        )