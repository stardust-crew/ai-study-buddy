from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.tools.youtube import YouTubeTools
from agno.playground import Playground, serve_playground_app
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder
from agno.document.chunking.agentic import AgenticChunking
import re
from typing import List
from pydantic import BaseModel, Field
import json
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

# def evaluate_answer(question, user_answer, correct_answer):
#     prompt = f"Question: {question}\nUser's answer: {user_answer}\nCorrect answer: {correct_answer}\nEvaluate if the user's answer is correct and provide an explanation."
#     response = quiz.run_response(prompt)
#     return response.content

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
                embedder=GeminiEmbedder(api_key="AIzaSyDaUUXY0_H8kWt8068ew2Tu_95wLYOXByE"),
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
            model=Gemini(id="gemini-2.0-flash", api_key="AIzaSyDaUUXY0_H8kWt8068ew2Tu_95wLYOXByE"),
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

# print(dict(generate_quiz('pca'))['quiz'][0])

def test_pdf_agent():
    # Path to a local PDF file
    pdf_path = Path("PCA.pdf")
    
    # Read the PDF file and convert to bytes IO (mimicking file upload)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Create a BytesIO object to simulate an uploaded file
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_file.name = pdf_path.name  # Add name attribute to mimic uploaded file
    
    # Initialize an agent with the PDF
    agent = initialize_agent_with_pdf(
        pdf_file=pdf_file,
        agent_name="TestAgent",
        agent_role="test assistant",
        table_name="test_pdf"
    )
    
    # Test the agent by generating a quiz
    topic = "PCA"
    print(f"Generating quiz about: {topic}")
    
    quiz_response = generate_quiz(agent, topic, num_questions=5)
    
    # Print the generated quiz
    for i, question in enumerate(quiz_response.quiz, 1):
        print(f"\nQuestion {i}: {question.question}")
        for j, option in enumerate(question.options):
            print(f"  {chr(65+j)}) {option}")
        print(f"Correct answer: {chr(65+question.correct)}")
    
    return agent

# Run the test
if __name__ == "__main__":
    test_agent = test_pdf_agent()

# app = Playground(agents=[quiz]).get_app()
# if __name__ == "__main__":
#     serve_playground_app("evaluation:quiz", reload=True)
