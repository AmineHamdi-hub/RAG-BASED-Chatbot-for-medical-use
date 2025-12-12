from src.rag import retrieve_similar
from src.llm_wrapper import GroqLLM

class MedicalRAGAgent:
    def __init__(self, llm=None):
        self.llm = llm or GroqLLM()

    def answer(self, query: str) -> str:
        """Runs RAG retrieval + always answers, even with no/weak context."""
        
        retrieved = retrieve_similar(query)
        context = "\n".join(retrieved) if retrieved else "No relevant context was found."

        prompt = (
            "You are a medical assistant. "
            "Use the context below ONLY if it is relevant. "
            "If the context is empty or irrelevant, rely on general medical knowledge. "
            "Always provide an answer, even if the context is insufficient.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer clearly and helpfully:"
        )

        return self.llm(prompt)
