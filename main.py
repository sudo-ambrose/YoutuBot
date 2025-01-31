##IMPORTS
import gradio as gr
from agent import agent
from functions import extract_agent_output

##GRADIO INTERFACE

# Define the chat function
def chat_with_agent(message, history):
    """
    Handles the interaction with the agent.
    
    Args:
        message (str): User's input message.
        history (list): Chat history with role and content keys.
        
    Returns:
        str: Agent's response to the user's input.
    """
    try:
        # Get the agent's response
        response = extract_agent_output(agent(message))
    except Exception as e:
        response = f"Error: {str(e)}"
    
    return response  # Only return the chatbot's response, not the history

# Example questions to display
example_questions = [
    "Helloo! What can you do??",
    "I love you.",
    "Let's get started!"
]

# Create the Gradio interface
chat_interface = gr.ChatInterface(
    fn=chat_with_agent,  # Link the chat function
    type="messages",     # Use OpenAI-style messages for history
    title="YoutuBot",  # Set a title for the interface
    description="Chat with YoutuBot. Just submit a link and ask questions!",
    examples=example_questions
)

# Launch the interface
chat_interface.launch(share=True)