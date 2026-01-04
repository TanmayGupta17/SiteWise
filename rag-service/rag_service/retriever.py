"""
Question answering system that retrieves relevant content and generates answers.
This is the final step of our RAG pipeline - answering questions.
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
import os
from .config import (
    DEFAULT_TOP_K, SIMILARITY_THRESHOLD, MAX_ANSWER_LENGTH, REFUSAL_PHRASES
)
from .indexer import TextIndexer, DocumentChunk
from .utils import logger

# Configure Gemini globally once (top of file)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# for model in genai.list_models():
#     if "generateContent" in model.supported_generation_methods:
#         print(model.name)

@dataclass
class SourceInfo:
    """Information about where an answer came from."""
    url: str
    snippet: str
    similarity_score: float

@dataclass
class TimingInfo:
    """Performance timing information."""
    retrieval_ms: float
    generation_ms: float
    total_ms: float

@dataclass
class QAResponse:
    """Complete response to a user question."""
    answer: str
    sources: List[SourceInfo]
    timings: TimingInfo
    is_refusal: bool = False

class GroundedQA:
    """
    Question answering system that ONLY answers from retrieved content.
    This prevents hallucination - if we don't have the info, we say so.
    """
    
    def __init__(self, indexer: TextIndexer):
        self.indexer = indexer
        self.embedding_model = indexer.embedding_model
        
        # Simple templates for generating answers
        # In production, you'd replace this with a proper LLM like Ollama
        self.answer_templates = {
            'not_found': "I don't have enough information in the crawled content to answer this question.",
            'insufficient': "The crawled content doesn't contain sufficient information about this topic."
        }
    
    def _retrieve_chunks(self, question: str, top_k: int = DEFAULT_TOP_K) -> Tuple[List[DocumentChunk], List[float], float]:
        """
        Find the most relevant chunks for answering the question.
        This is the 'Retrieval' part of RAG.
        """
        start_time = time.time()
        
        if not self.indexer.faiss_index or not self.indexer.document_chunks:
            logger.warning("Index not loaded or empty")
            return [], [], 0.0
        
        try:
            # Convert the question to an embedding vector
            question_embedding = self.embedding_model.encode(
                [question], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            if question_embedding.ndim == 1:
                question_embedding = question_embedding.reshape(1, -1)
            
            k = min(top_k, len(self.indexer.document_chunks))
            # Search for similar chunks in our index
            similarities, indices = self.indexer.faiss_index.search(
                question_embedding.astype(np.float32), 
                k
            )
            # print(similarities,indices, "hi")
            # Filter results by similarity threshold
            relevant_chunks = []
            similarity_scores = []
            
            print(f"[DEBUG] Found {len(similarities[0])} results from FAISS")
            for similarity, idx in zip(similarities[0], indices[0]):
                print(f"[DEBUG] Chunk {idx}: similarity={similarity:.4f}, threshold={SIMILARITY_THRESHOLD}")
                if idx >= 0 and similarity >= SIMILARITY_THRESHOLD:
                    chunk = self.indexer.document_chunks[idx]
                    relevant_chunks.append(chunk)
                    similarity_scores.append(float(similarity))
                    print(f"[DEBUG] ✓ Accepted chunk {idx} with score {similarity:.4f}")
                else:
                    print(f"[DEBUG] ✗ Rejected chunk {idx} (too low or invalid)")
            
            retrieval_time = (time.time() - start_time) * 1000
            
            
            logger.info("Retrieved chunks", 
                       question_length=len(question),
                       chunks_found=len(relevant_chunks),
                       top_similarity=max(similarity_scores) if similarity_scores else 0,
                       retrieval_time_ms=retrieval_time)
            print(relevant_chunks,similarity_scores,retrieval_time,"hello")
            
            return relevant_chunks, similarity_scores, retrieval_time
            
        except Exception as e:
            logger.error("Error during retrieval", error=str(e))
            return [], [], (time.time() - start_time) * 1000
    
    def _build_context(self, chunks: List[DocumentChunk], max_length: int = 4000) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            # Format each chunk with source info
            source_info = f"Source: {chunk.url}\n"
            content_info = f"Content: {chunk.content}\n\n"
            
            chunk_text = source_info + content_info
            
            # Check if we have room for this chunk
            if current_length + len(chunk_text) > max_length:
                # Try to fit a truncated version
                remaining_space = max_length - current_length - len(source_info) - 20
                if remaining_space > 100:  # Only if we have meaningful space
                    truncated_content = chunk.content[:remaining_space] + "..."
                    chunk_text = source_info + f"Content: {truncated_content}\n\n"
                else:
                    break  # No more room
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    # def _generate_answer(self, question: str, context: str) -> Tuple[str, float]:
    #     """
    #     Generate an answer from the context.
        
    #     NOTE: This is currently a simple rule-based system.
    #     In production, replace this with a local LLM like:
    #     - Ollama (llama2, mistral, etc.)
    #     - GPT4All
    #     - Any local model via Hugging Face transformers
    #     """
    #     start_time = time.time()
        
    #     if not context.strip():
    #         answer = self.answer_templates['not_found']
    #         generation_time = (time.time() - start_time) * 1000
    #         return answer, generation_time
        
    #     # Simple rule-based answer generation
    #     question_lower = question.lower()
        
    #     # Extract relevant sentences from context
    #     sentences = []
    #     for line in context.split('\n'):
    #         if line.startswith('Content:'):
    #             content = line[8:].strip()  # Remove 'Content:' prefix
    #             sentences.extend([s.strip() for s in content.split('.') if s.strip()])
        
    #     # Select most relevant sentences based on question type
    #     if any(word in question_lower for word in ['what is', 'what are', 'define']):
    #         # Definition questions - take first few sentences
    #         relevant_sentences = sentences[:3]
    #     elif any(word in question_lower for word in ['how', 'why', 'when', 'where']):
    #         # Explanation questions - take more context
    #         relevant_sentences = sentences[:4]
    #     else:
    #         # General questions
    #         relevant_sentences = sentences[:3]
        
    #     if not relevant_sentences:
    #         answer = self.answer_templates['insufficient']
    #     else:
    #         # Combine sentences into an answer
    #         answer_text = '. '.join(relevant_sentences)
    #         if answer_text and not answer_text.endswith('.'):
    #             answer_text += '.'
            
    #         # Ensure answer isn't too long
    #         if len(answer_text) > MAX_ANSWER_LENGTH:
    #             answer_text = answer_text[:MAX_ANSWER_LENGTH-3] + "..."
            
    #         # Add context attribution
    #         answer = f"Based on the crawled content: {answer_text}"
        
    #     generation_time = (time.time() - start_time) * 1000
        
    #     logger.info("Generated answer", 
    #                answer_length=len(answer),
    #                generation_time_ms=generation_time)
        
    #     return answer, generation_time

    def _generate_answer(self, question: str, context: str) -> Tuple[str, float]:
        """
        Generate an answer using Google's Gemini API, based on the provided question and context.
        """
        start_time = time.time()

        # If context is empty, return fallback response
        if not context.strip():
            answer = self.answer_templates['not_found']
            generation_time = (time.time() - start_time) * 1000
            return answer, generation_time

        try:
            # Prepare the prompt for Gemini
            prompt = f"""
        You are a helpful assistant that answers strictly based on the provided context.
        If the context does not have enough information to answer the question, respond with:
        "I don't have enough information in the crawled content to answer this question."

        Context:
        {context}

        Question: {question}

        Answer:
        """

            # Initialize model (you can use gemini-1.5-pro for higher quality)
            model = genai.GenerativeModel("gemini-2.5-flash-lite")

            # Generate answer
            response = model.generate_content(prompt)

            # Extract text safely
            answer = response.text.strip() if response and response.text else self.answer_templates["insufficient"]

            # Handle overly long answers
            if len(answer) > MAX_ANSWER_LENGTH:
                answer = answer[:MAX_ANSWER_LENGTH - 3] + "..."

            generation_time = (time.time() - start_time) * 1000

            print(f"[INFO] Generated answer with Gemini (time={generation_time:.2f}ms)")
            return answer, generation_time

        except Exception as e:
            print(f"[ERROR] Gemini generation failed: {e}")
            answer = self.answer_templates["insufficient"]
            generation_time = (time.time() - start_time) * 1000
            return answer, generation_time
        
    def _check_safety(self, question: str, context: str) -> bool:
        """
        Check for potential prompt injection or unsafe content.
        This is crucial for production systems.
        """
        # Common prompt injection patterns
        dangerous_patterns = [
            'ignore previous instructions',
            'forget everything',
            'new instructions:',
            'system prompt',
            'you are now',
            'roleplay as',
            'pretend to be',
            'act as if',
            'disregard',
            'override'
        ]
        
        combined_text = (question + " " + context).lower()
        
        for pattern in dangerous_patterns:
            if pattern in combined_text:
                logger.warning("Potential prompt injection detected", 
                             pattern=pattern,
                             question_preview=question[:100])
                return False
        
        return True
    
    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> QAResponse:
        """
        Main method to answer a question.
        This orchestrates the entire RAG process.
        """
        total_start_time = time.time()
        
        logger.info("Processing question", 
                   question=question[:100] + "..." if len(question) > 100 else question,
                   top_k=top_k)
        
        try:
            # Step 1: Retrieve relevant chunks
            chunks, similarities, retrieval_time = self._retrieve_chunks(question, top_k)
            logger.info("Chunks retrieved", count=len(chunks))
            logger.info("Similarities", scores=similarities)
            
            # Step 2: Build context from chunks
            context = self._build_context(chunks)
            
            # Step 3: Safety check
            if not self._check_safety(question, context):
                return QAResponse(
                    answer="I cannot process this request due to safety concerns.",
                    sources=[],
                    timings=TimingInfo(retrieval_time, 0, (time.time() - total_start_time) * 1000),
                    is_refusal=True
                )
            
            # Step 4: Generate answer
            answer, generation_time = self._generate_answer(question, context)
            
            # Step 5: Check if this is a refusal
            is_refusal = any(phrase in answer for phrase in REFUSAL_PHRASES)
            
            # Step 6: Create source information
            sources = []
            for chunk, similarity in zip(chunks, similarities):
                # Create a snippet (first 200 chars of chunk)
                snippet = chunk.content[:200]
                if len(chunk.content) > 200:
                    snippet += "..."
                
                source = SourceInfo(
                    url=chunk.url,
                    snippet=snippet,
                    similarity_score=similarity
                )
                sources.append(source)
            
            total_time = (time.time() - total_start_time) * 1000
            
            response = QAResponse(
                answer=answer,
                sources=sources,
                timings=TimingInfo(
                    retrieval_ms=retrieval_time,
                    generation_ms=generation_time,
                    total_ms=total_time
                ),
                is_refusal=is_refusal
            )
            
            logger.info("Question answered",
                       is_refusal=is_refusal,
                       sources_count=len(sources),
                       total_time_ms=total_time)
            
            return response
            
        except Exception as e:
            logger.error("Error answering question", error=str(e))
            
            return QAResponse(
                answer="An error occurred while processing your question.",
                sources=[],
                timings=TimingInfo(0, 0, (time.time() - total_start_time) * 1000),
                is_refusal=True
            )