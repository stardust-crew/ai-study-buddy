import streamlit as st
import pandas as pd
from datetime import datetime
import random
import io

# Import agent functions
from quiz_agent import initialize_agent_with_pdf, generate_quiz
from chat_agent import initialize_chat_with_pdf

# Page configuration
st.set_page_config(
    page_title="EduChat AI Learning Platform",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if 'pdf_data' not in st.session_state:
    # Store all PDF-specific data in this dictionary
    # Structure: { 'pdf_name': {'agent': agent_obj, 'file': file_obj, 'chat_history': [], 'quiz_state': {}} }
    st.session_state.pdf_data = {}

if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None

# Function to get AI response using study_partner agent
def get_ai_response(message, topic=None):
    if not topic:
        return "Please upload or select notes first."
    
    pdf_data = st.session_state.pdf_data.get(topic)
    if not pdf_data or 'chat' not in pdf_data:
        return "Chat agent not initialized for this topic."
    
    chat_agent = pdf_data['chat']
    response = chat_agent.run(message)
    return response.content if hasattr(response, 'content') else str(response)

# Function to handle quiz submission
def submit_quiz(topic):
    if not topic:
        return
    
    pdf_data = st.session_state.pdf_data[topic]
    quiz_state = pdf_data['quiz_state']
    
    score = 0
    for i, answer in enumerate(quiz_state['answers']):
        if answer == quiz_state['questions'][i]["correct"]:
            score += 1
    
    quiz_state['score'] = score
    quiz_state['total'] = len(quiz_state['questions'])
    quiz_state['active'] = False

# Main app UI
st.title("📚 EduChat AI Learning Platform")

# Sidebar for navigation and file uploads
with st.sidebar:
    st.header("Your Learning Materials")
    
    # File uploader for PDFs
    uploaded_file = st.file_uploader("Upload PDF for study and quizzes", type=["pdf"])
    
    if uploaded_file is not None:
        # Create BytesIO object from the uploaded file
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        pdf_file.name = uploaded_file.name  # Add name attribute
        
        topic_name = uploaded_file.name.split(".")[0]
        
        # Check if this PDF was already uploaded
        if topic_name not in st.session_state.pdf_data:
            # Initialize PDF agent with the uploaded file
            with st.spinner("Processing PDF..."):
                try:
                    pdf_agent = initialize_agent_with_pdf(
                        pdf_file=pdf_file,
                        agent_name="StudyScout",
                        agent_role="study assistant",
                        table_name=topic_name.lower().replace(" ", "_")
                    )
                    
                    chat_agent = initialize_chat_with_pdf(
                        pdf_file=pdf_file,
                        table_name=topic_name.lower().replace(" ", "_")
                    )

                    # Create a new entry for this PDF
                    st.session_state.pdf_data[topic_name] = {
                        'agent': pdf_agent,
                        'chat': chat_agent,
                        'file': pdf_file,
                        'chat_history': [],
                        'quiz_state': {
                            'active': False,
                            'questions': [],
                            'answers': [],
                            'score': 0,
                            'total': 0,
                            'custom_topic': ""
                        }
                    }
                    
                    st.session_state.current_topic = topic_name
                    st.success(f"'{topic_name}' PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        else:
            st.info(f"'{topic_name}' is already in your library.")
            st.session_state.current_topic = topic_name
    
    st.divider()
    
    # Display list of available PDFs
    st.subheader("Your PDFs")
    if st.session_state.pdf_data:
        for topic in st.session_state.pdf_data.keys():
            is_selected = topic == st.session_state.current_topic
            button_text = f"📝 {topic}" + (" ✓" if is_selected else "")
            if st.button(button_text, key=f"select_{topic}"):
                st.session_state.current_topic = topic
                st.rerun()
    else:
        st.info("No PDFs uploaded yet. Upload a PDF to get started!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.current_topic:
        topic = st.session_state.current_topic
        pdf_data = st.session_state.pdf_data[topic]
        quiz_state = pdf_data['quiz_state']
        
        st.header(f"Learning Assistant: {topic}")
        
        # Create tabs for chat and quiz
        chat_tab, quiz_tab = st.tabs(["Chat", "Quiz"])
        
        # Chat interface tab
        with chat_tab:
            # Display chat history
            for message in pdf_data['chat_history']:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        # Use markdown to render links as clickable
                        st.markdown(message["content"])
            
            # Chat input
            user_message = st.chat_input(f"Ask about {topic}...")
            if user_message:
                st.chat_message("user").write(user_message)
                pdf_data['chat_history'].append({"role": "user", "content": user_message})
                
                # Get AI response
                with st.spinner("Thinking..."):
                    response = get_ai_response(user_message, topic)
                
                st.chat_message("assistant").write(response)
                pdf_data['chat_history'].append({"role": "assistant", "content": response})
        
        # Quiz interface tab
        with quiz_tab:
            if not quiz_state['active']:
                st.subheader("Generate Quiz")
                
                # Input for custom topic
                quiz_state['custom_topic'] = st.text_input(
                    "Enter a specific topic for the quiz (leave empty to use the whole document):",
                    value=quiz_state.get('custom_topic', ""),
                    key=f"custom_topic_{topic}"
                )
                
                num_questions = st.slider("Number of questions:", min_value=1, max_value=10, value=5)
                
                # Quiz generation button
                if st.button("Generate Quiz", key=f"gen_quiz_{topic}"):
                    with st.spinner("Generating quiz questions..."):
                        quiz_topic = quiz_state['custom_topic'] if quiz_state['custom_topic'] else topic
                        try:
                            quiz_response = generate_quiz(pdf_data['agent'], quiz_topic, num_questions)
                            
                            # Convert Pydantic quiz to format needed for UI
                            questions = []
                            for q in quiz_response.quiz:
                                questions.append({
                                    "question": q.question,
                                    "options": q.options,
                                    "correct": q.correct
                                })
                            
                            quiz_state['questions'] = questions
                            quiz_state['answers'] = [None] * len(questions)
                            quiz_state['active'] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating quiz: {str(e)}")
                
                # Display previous quiz results if available
                if quiz_state['total'] > 0:
                    st.success(f"Previous quiz score: {quiz_state['score']}/{quiz_state['total']}")
                    
                    # Display correct answers from previous quiz
                    with st.expander("Review Previous Quiz"):
                        for i, question in enumerate(quiz_state['questions']):
                            st.write(f"**Question {i+1}:** {question['question']}")
                            st.write(f"Your answer: {question['options'][quiz_state['answers'][i]]}")
                            st.write(f"Correct answer: {question['options'][question['correct']]}")
                            if quiz_state['answers'][i] == question['correct']:
                                st.success("Correct! ✓")
                            else:
                                st.error("Incorrect ✗")
                            st.divider()
            
            # Active quiz display
            else:
                quiz_topic = quiz_state['custom_topic'] if quiz_state['custom_topic'] else topic
                st.subheader(f"Quiz on {quiz_topic}")
                
                for i, question in enumerate(quiz_state['questions']):
                    st.write(f"**Question {i+1}:** {question['question']}")
                    quiz_state['answers'][i] = st.radio(
                        "Select your answer:",
                        options=range(len(question["options"])),
                        format_func=lambda x: question["options"][x],
                        key=f"q{topic}_{i}",
                        index=0 if quiz_state['answers'][i] is None else quiz_state['answers'][i]
                    )
                    st.divider()
                
                col_submit, col_cancel = st.columns([1, 5])
                with col_submit:
                    if st.button("Submit Quiz", key=f"submit_{topic}"):
                        submit_quiz(topic)
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel Quiz", key=f"cancel_{topic}"):
                        quiz_state['active'] = False
                        st.rerun()
    else:
        st.info("Please upload a PDF document to get started with your AI learning assistant!")

with col2:
    if st.session_state.current_topic:
        topic = st.session_state.current_topic
        pdf_data = st.session_state.pdf_data[topic]
        quiz_state = pdf_data['quiz_state']
        
        st.header("Document Information")
        
        # Display information about the PDF
        st.metric("Current PDF", topic)
        
        if quiz_state['custom_topic']:
            st.metric("Quiz Topic", quiz_state['custom_topic'])
            
        # Learning progress tracking section
        st.subheader("Learning Progress")
        
        # Display a progress bar for current topic comprehension (simplified)
        comprehension = min(0.1 * len(pdf_data['chat_history']), 1.0)
        st.write("Topic Comprehension")
        st.progress(comprehension)
        
        # Display quiz performance if available
        if quiz_state['total'] > 0:
            quiz_accuracy = quiz_state['score'] / quiz_state['total']
            st.write("Quiz Performance")
            st.progress(quiz_accuracy)
            
            # Generate some performance data
            dates = pd.date_range(end=datetime.now(), periods=5).tolist()
            performances = [random.uniform(0.6, 0.95) for _ in range(5)]
            performances[-1] = quiz_accuracy  # Set the latest performance to actual quiz result
            
            chart_data = pd.DataFrame({
                "date": dates,
                "performance": performances
            })
            
            st.line_chart(chart_data.set_index("date"))
            
        # Study resources section
        st.subheader("Additional Study Resources")
        st.write("Ask the chat assistant to find resources about:")
        
        # Suggested topics based on current PDF
        suggested_topics = [
            "Key concepts in " + topic,
            "Practical applications of " + topic,
            "YouTube tutorials on " + topic,
            "Create a study plan for " + topic
        ]
        
        for suggested_topic in suggested_topics:
            if st.button(suggested_topic, key=f"resource_{topic}_{suggested_topic}"):
                # Add this query to chat and get response
                pdf_data['chat_history'].append({"role": "user", "content": suggested_topic})
                with st.spinner("Searching for resources..."):
                    response = get_ai_response(suggested_topic, topic)
                pdf_data['chat_history'].append({"role": "assistant", "content": response})
                st.rerun()
    else:
        st.info("Upload a PDF document to get started with your AI learning assistant!")