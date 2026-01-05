# =============================================================================
# rag_pipeline.py - End-to-End RAG Question Answering
# =============================================================================
"""
This module implements the complete RAG (Retrieval-Augmented Generation) pipeline.

KEY CONCEPTS FOR BEGINNERS:

WHAT IS RAG?
RAG = Retrieval-Augmented Generation
It's a technique that combines:
1. RETRIEVAL: Find relevant documents from a knowledge base
2. AUGMENTATION: Add those documents as context to a prompt
3. GENERATION: Use an LLM to generate an answer based on the context

WHY RAG?
- LLMs have knowledge cutoffs (don't know recent info)
- LLMs can hallucinate (make up facts)
- RAG grounds the LLM in your actual data
- You can update the knowledge base without retraining

THE RAG PIPELINE:
1. User asks a question
2. Question is embedded (converted to vector)
3. Similar documents are retrieved from vector store
4. Documents are formatted into context
5. Context + question are sent to LLM
6. LLM generates answer based on context
7. Answer + sources are returned to user

         ┌─────────────┐
         │   Question  │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │   Embed     │
         └──────┬──────┘
                │
                ▼
    ┌───────────────────────┐
    │   Vector Store        │
    │   (FAISS)             │
    │   Find similar docs   │
    └───────────┬───────────┘
                │
                ▼
         ┌─────────────┐
         │  Retrieved  │
         │  Documents  │
         └──────┬──────┘
                │
                ▼
    ┌───────────────────────┐
    │   Format Prompt       │
    │   Context + Question  │
    └───────────┬───────────┘
                │
                ▼
         ┌─────────────┐
         │     LLM     │
         │  (FLAN-T5)  │
         └──────┬──────┘
                │
                ▼
    ┌───────────────────────┐
    │   Answer + Sources    │
    └───────────────────────┘
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document

from . import config
from .vectorstore import load_vector_store, get_retriever, search_similar
from .llm import get_llm, RAG_PROMPT_TEMPLATE, format_docs_for_context


@dataclass
class RAGResponse:
    """
    Container for RAG pipeline response.
    
    Attributes:
        question: The original question asked
        answer: The generated answer from the LLM
        sources: List of retrieved documents used as context
        context: The formatted context string sent to the LLM
    """
    question: str
    answer: str
    sources: List[Document]
    context: str


class RAGPipeline:
    """
    End-to-end RAG pipeline for answering questions about support tickets.
    
    This class encapsulates the entire RAG workflow:
    1. Load vector store and LLM
    2. Retrieve relevant documents for a question
    3. Generate answer using LLM with retrieved context
    4. Return answer with source documents
    
    Example:
        >>> rag = RAGPipeline()
        >>> response = rag.ask("What are common billing issues?")
        >>> print(response.answer)
        >>> for doc in response.sources:
        ...     print(doc.metadata['ticket_id'])
    """
    
    def __init__(
        self,
        vectorstore=None,
        llm=None,
        retrieval_k: int = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vectorstore: Pre-loaded vector store (loads from disk if None)
            llm: Pre-loaded LLM (loads FLAN-T5 if None)
            retrieval_k: Number of documents to retrieve (default from config)
        """
        print("=" * 50)
        print("Initializing RAG Pipeline...")
        print("=" * 50)
        
        # Set retrieval k
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K
        
        # Load vector store
        if vectorstore is None:
            print("\nStep 1: Loading vector store...")
            self.vectorstore = load_vector_store()
        else:
            self.vectorstore = vectorstore
            print("✓ Using provided vector store")
        
        # Load LLM
        if llm is None:
            print("\nStep 2: Loading LLM...")
            self.llm = get_llm()
        else:
            self.llm = llm
            print("✓ Using provided LLM")
        
        # Create retriever
        print("\nStep 3: Creating retriever...")
        self.retriever = get_retriever(self.vectorstore, k=self.retrieval_k)
        
        print("\n" + "=" * 50)
        print("✓ RAG Pipeline ready!")
        print("=" * 50)
    
    def retrieve(self, question: str) -> List[Document]:
        """
        Retrieve relevant documents for a question.
        
        This is the "R" in RAG - Retrieval.
        
        Args:
            question: The user's question
            
        Returns:
            List of relevant Documents
        """
        docs = self.retriever.invoke(question)
        return docs
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using the LLM.
        
        This is the "G" in RAG - Generation.
        
        Args:
            question: The user's question
            context: Formatted context from retrieved documents
            
        Returns:
            Generated answer string
        """
        # Format the prompt with context and question
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Generate answer
        answer = self.llm.invoke(prompt)
        
        return answer.strip()
    
    def ask(self, question: str) -> RAGResponse:
        """
        Ask a question and get an answer with sources.
        
        This is the main method that runs the full RAG pipeline:
        1. Retrieve relevant documents
        2. Format context
        3. Generate answer
        4. Return answer + sources
        
        Args:
            question: The user's question
            
        Returns:
            RAGResponse with answer and sources
            
        Example:
            >>> response = rag.ask("What products have the most issues?")
            >>> print(response.answer)
        """
        # Step 1: Retrieve relevant documents
        sources = self.retrieve(question)
        
        # Step 2: Format documents into context
        context = format_docs_for_context(sources)
        
        # Step 3: Generate answer
        answer = self.generate(question, context)
        
        # Step 4: Return response with sources
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context
        )
    
    def ask_with_details(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get detailed response as dictionary.
        
        Useful for debugging and evaluation.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with all details
        """
        response = self.ask(question)
        
        return {
            "question": response.question,
            "answer": response.answer,
            "num_sources": len(response.sources),
            "sources": [
                {
                    "ticket_id": doc.metadata.get("ticket_id"),
                    "product": doc.metadata.get("product"),
                    "ticket_type": doc.metadata.get("ticket_type"),
                    "priority": doc.metadata.get("priority"),
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in response.sources
            ],
            "context_length": len(response.context),
        }


def print_rag_response(response: RAGResponse) -> None:
    """
    Pretty print a RAG response.
    
    Args:
        response: RAGResponse object to display
    """
    print("=" * 70)
    print("QUESTION:")
    print(response.question)
    print("=" * 70)
    
    print("\nANSWER:")
    print(response.answer)
    
    print("\n" + "-" * 70)
    print(f"SOURCES ({len(response.sources)} documents retrieved):")
    print("-" * 70)
    
    for i, doc in enumerate(response.sources, 1):
        print(f"\n📄 Source {i}:")
        print(f"   Ticket ID: {doc.metadata.get('ticket_id', 'N/A')}")
        print(f"   Product: {doc.metadata.get('product', 'N/A')}")
        print(f"   Type: {doc.metadata.get('ticket_type', 'N/A')}")
        print(f"   Priority: {doc.metadata.get('priority', 'N/A')}")
        print(f"   Content: {doc.page_content[:150]}...")
    
    print("\n" + "=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_ask(question: str, k: int = 5) -> RAGResponse:
    """
    Quick way to ask a question without manually initializing pipeline.
    
    Note: This creates a new pipeline each time, which is slow.
    For multiple questions, create a RAGPipeline instance and reuse it.
    
    Args:
        question: The question to ask
        k: Number of documents to retrieve
        
    Returns:
        RAGResponse with answer and sources
    """
    rag = RAGPipeline(retrieval_k=k)
    return rag.ask(question)
