from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.tools.youtube import YouTubeTools
from agno.tools.tavily import TavilyTools
from agno.tools.todoist import TodoistTools
from agno.models.groq import Groq
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.embedder.google import GeminiEmbedder
from agno.document.chunking.agentic import AgenticChunking
from agno.vectordb.lancedb import LanceDb

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("gro_api_key")
GOOGLE_API_KEY = os.getenv("google_api_key")
TAVILY_API_KEY = os.getenv("tavily_api_key")
TODOIST_TOKEN = os.getenv("todoist_token")

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
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    tools=[TodoistTools(api_token=TODOIST_TOKEN)],
    markdown=True,
    show_tool_calls=True,
    expected_output="Todoist task created successfully.",
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
            vector_db = LanceDb(
                uri='./tmp/lancedb',
                table_name=table_name,
                embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY),
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
            model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
            tools=[TavilyTools(api_key=TAVILY_API_KEY), YouTubeTools()],
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
