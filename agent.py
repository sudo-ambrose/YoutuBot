##IMPORTS
import functions
from variables import OPENAI_API_KEY
from functions import tools
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4o',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    k=7,
    return_messages=True
)

# custom prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a helpful assistant specialized in processing YouTube video transcripts. Your role is to fetch, process, and retrieve data exclusively from these transcripts to answer user questions. 

    **Rules and Workflow:**
    1. **Processing New Videos:**
       - Upon receiving a URL, use a tool to fetch, process, and index the transcript from the video.
       - After completing this process, inform the user that the transcript has been succesfully indexed and that you are ready for questions" 
       - Do not generate any questions or provide speculative information.

    2. **Answering Questions:**
       - When the user asks a question, retrieve the most relevant data using a tool.
       - You can customize the query you pass to the GetRelevantData tool in order to perfom a better similarity search. 
       - Your responses must exclusively rely on data returned by the tool. 
       - Do not use your reasoning, beyond the tool's output to address any question.
    
    3. **Resetting for a New Video:**
       - Each time a new URL is provided, previous data is no longer accessible and you start from scratch.
       - Ensure the new transcript is fetched, processed, and indexed before answering any subsequent questions.

    **Critical Instructions:**
    - Always use tools for answering questions and never rely on your reasoning.
    - Do not respond to questions without an explicit user query or tool-generated data.
    - Never generate questions or make assumptions; only respond to direct user inputs.

    **Memory Context:**
    - Here is the previous conversation: `{chat_history}`

    Follow these guidelines strictly to ensure accurate, tool-based assistance for the user.
    """
)

# Initialize the agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=conversational_memory,
    #system_prompt=chat_prompt,
    early_stopping_method='generate',
    handle_parsing_errors=True,
    max_iterations=3,
    agent_kwargs={
        "system_message": system_prompt,
        "memory_key": "chat_history"
    }
)