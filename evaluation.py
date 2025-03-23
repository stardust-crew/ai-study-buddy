from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder
from agno.document.chunking.agentic import AgenticChunking
from typing import List
from pydantic import BaseModel, Field
import io
from pathlib import Path

class quiz_ques(BaseModel):
    question: str =  Field(..., description="The question text")
    options: List[str] = Field(..., description="options for the quiz Option A Option B Option C Option D")
    correct: int = Field(..., description="index for the correct ans")

class Quiz(BaseModel):
    quiz: List[quiz_ques] = Field(..., description="questions of the quiz")



def generate_quiz(agent, topic, num_questions=5):
    prompt = f"""
    Please generate {num_questions} multiple-choice quiz questions about {topic}.
    The questions should be based on the information in the knowledge base.
    Make sure each question has 4 options and marks the correct answer.
    """
    
    # Get the response as a Pydantic model
    response = agent.run(prompt)
    
    # Extract the quiz questions from the response
    return response.content

def initialize_agent_with_pdf(pdf_file, agent_name="StudyScout", agent_role="study assistant", table_name=None):
    import tempfile
    import os
    import uuid
    
    # Create a unique table name if not provided
    if table_name is None:
        table_name = f"pdf_{uuid.uuid4().hex[:8]}"
    else:
        # Create a PostgreSQL-safe table name
        table_name = f"pdf_{table_name.lower().replace(' ', '_').replace('-', '_')}"
    
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        # Create a knowledge base for this specific PDF
        custom_kb = PDFKnowledgeBase(
            path=pdf_path,
            vector_db=PgVector(
                table_name=table_name,
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=GeminiEmbedder(api_key=""),
            ),
            reader=PDFReader(chunk=True, chunking_strategy=AgenticChunking()),
        )
        
        # Load the document into the vector DB
        custom_kb.load(recreate=False)
        
        # Initialize a new agent with this knowledge base
        agent = Agent(
            name=agent_name,
            knowledge=custom_kb,
            search_knowledge=True,
            role=agent_role,
            model=Gemini(id="gemini-2.0-flash", api_key=""),
            markdown=True,
            description="you are a study partner who assists users in finding resources and make quizes on various topics.",
            instructions=[
                "Use the knowledge base to answer questions about the PDF content",
                "make quizes on the given topic from the knowledge base",
                "evaluate the answers of the user from the knowledge base",
                "provide explanations on the answers",
            ],
            response_model=Quiz,
        )
        
        os.unlink(pdf_path)
        
        return agent
    except Exception as e:
        os.unlink(pdf_path)
        raise e
