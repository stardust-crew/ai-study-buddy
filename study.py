from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.tools.youtube import YouTubeTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from agno.tools.todoist import TodoistTools
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder
from agno.document.chunking.agentic import AgenticChunking

from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel
from rich.json import JSON
import json

todoist_agent = Agent(
    name="Todoist Agent",
    role="Manage your todoist tasks",
    instructions=[
        "When given a task, create a todoist task for it.",
        "When given a list of tasks, create a todoist task for each one.",
        "When given a task to update, update the todoist task.",
        "When given a task to delete, delete the todoist task.",
        "When given a task to get, get the todoist task.",
    ],
    agent_id="todoist-agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key="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"),
    tools=[TodoistTools(api_token="4545fbcc373c320d46b7b0d53ca389e6b5f96e43")],
    markdown=True,
    show_tool_calls=True,
    expected_output="Todoist task created successfully.",
)



study_partner = Agent(
    name="StudyScout",
    role="collect resources, make study plans, and provide explanations",
    team=[todoist_agent],
    model=Gemini(id="gemini-2.0-flash", api_key="AIzaSyDaUUXY0_H8kWt8068ew2Tu_95wLYOXByE"),
    tools=[TavilyTools(api_key="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"), YouTubeTools()],
    markdown=True,
    description="You are a study partner who assists users in finding resources, answering questions, and providing explanations on various topics.",
    instructions=[
        """Use Tavily to search for relevant information on the given topic and verify information from multiple reliable sources.,
        Break down complex topics into digestible chunks and provide step-by-step explanations with practical examples.,
        Share curated learning resources including documentation, tutorials, articles, research papers, and community discussions.,
        Recommend high-quality YouTube videos and online courses that match the user's learning style and proficiency level.,
        Suggest hands-on projects and exercises to reinforce learning, ranging from beginner to advanced difficulty.,
        Create personalized study plans with clear milestones, deadlines, and progress tracking.,
        Provide tips for effective learning techniques, time management, and maintaining motivation.,
        Recommend relevant communities, forums, and study groups for peer learning and networking.,
        make a todoist list for the user to follow - give a list of tasks (daily tasks as separate function calls) to the todoist agent""",
    ],
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses=3,
    show_tool_calls=True
)

def initialize_chat_with_pdf(pdf_file, agent_name="StudyScout", agent_role="collect resources, make study plans, and provide explanations", table_name=None):
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
            name="StudyScout",
            knowledge=custom_kb,
            search_knowledge=True,
            add_references=True,
            role="collect resources, make study plans, and provide explanations",
            team=[todoist_agent],
            model=Gemini(id="gemini-2.0-flash", api_key="AIzaSyDaUUXY0_H8kWt8068ew2Tu_95wLYOXByE"),
            tools=[TavilyTools(api_key="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"), YouTubeTools()],
            markdown=True,
            description="You are a study partner who assists users in finding resources, answering questions, and providing explanations on various topics.",
            instructions=[
                """Use Tavily to search for relevant information on the given topic and verify information from multiple reliable sources.,
                Break down complex topics into digestible chunks and provide step-by-step explanations with practical examples.,
                Share curated learning resources including documentation, tutorials, articles, research papers, and community discussions.,
                Recommend high-quality YouTube videos and online courses that match the user's learning style and proficiency level.,
                Suggest hands-on projects and exercises to reinforce learning, ranging from beginner to advanced difficulty.,
                Create personalized study plans with clear milestones, deadlines, and progress tracking.,
                Provide tips for effective learning techniques, time management, and maintaining motivation.,
                Recommend relevant communities, forums, and study groups for peer learning and networking.,
                make a todoist list for the user to follow - give a list of tasks (daily tasks as separate function calls) to the todoist agent""",
            ],
            read_chat_history=True,
            add_history_to_messages=True,
            num_history_responses=3,
            show_tool_calls=True
        )
        
        os.unlink(pdf_path)
        
        return agent
    except Exception as e:
        os.unlink(pdf_path)
        raise e
