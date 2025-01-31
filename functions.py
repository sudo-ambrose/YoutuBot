##IMPORTS
import variables
from variables import tokenizer, index, index_name, embed, pc, vectorstore
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool

# Define the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Function to clear index
def reset_index():
    """Deletes all data in the Pinecone index."""
    try:
        # Delete all vectors in the index
        index.delete(delete_all=True)
        print(f"Index '{index_name}' has been cleared.")
    except Exception as e:
        print(f"Error while clearing the index: {e}")

# Function to get Youtube video transcript and meta data
def get_transcript_data_from_url(url):
    """
    Fetches the transcript of a YouTube video as both its original structure
    and a joined single string.
    
    Args:
    - url (str): YouTube video URL.

    Returns:
    - dict: Contains 'original' (list of dicts) and 'joined' (string).
    """
    # Extract the video ID using regex
    match = re.search(r"(?<=v=)[^&]+", url)
    if not match:
        raise ValueError("Invalid YouTube URL. Video ID not found.")
    
    video_id = match.group(0)
    
    try:
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine all text into a single string
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return {'original': transcript, 'joined': transcript_text, 'video_id': video_id}
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")


# Function to process and index the transcript given from the previous function
def process_and_index_transcript(transcript_data):
    """
    Processes the transcript by splitting into chunks, embedding, and indexing in Pinecone.
    
    Args:
    - transcript_data (dict): Contains 'original', 'joined', and 'video_id'.

    Returns:
    - None
    """
    transcript_text = transcript_data['joined']
    video_id = transcript_data['video_id']
    original_transcript = transcript_data['original']

    # Step 1: Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Define chunk size
        chunk_overlap=20,  # Define the overlap between chunks
        length_function=tiktoken_len,  # Use tiktoken length function
        separators=["\n\n", "\n", " ", ""]  # Define separators for splitting
    )
    chunks = text_splitter.split_text(transcript_text)

    # Step 2: Embed each chunk
    embeddings = embed.embed_documents(chunks)

    # Step 3: Create metadata for each chunk, including timestamps
    metadata = []
    for i, chunk in enumerate(chunks):
        # Find the original start time and duration for each chunk
        chunk_metadata = {
            "chunk_text": chunk,
            "video_id": video_id,
            "chunk_index": i,
            "start_time": original_transcript[i]["start"] if i < len(original_transcript) else None,
            "duration": original_transcript[i]["duration"] if i < len(original_transcript) else None,
        }
        metadata.append(chunk_metadata)

    # Step 4: Prepare vectors for Pinecone
    vectors = [
        {"id": f"{video_id}-chunk-{i}", "values": embeddings[i], "metadata": metadata[i]}
        for i in range(len(chunks))
    ]

    # Step 5: Connect to Pinecone and upsert vectors
    index = pc.Index(index_name)
    index.upsert(vectors)

    print(f"Successfully indexed video: {video_id} with {len(chunks)} chunks.")

# Function to extract the agent message as a string
def extract_agent_output(agent_response):
    """
    Extracts the 'output' field from the agent's response dictionary.
    
    Args:
        agent_response (dict): The agent's response containing the output.
        
    Returns:
        str: The extracted output text or an error message if not found.
    """
    try:
        # Check if 'output' key exists and return its value
        if "output" in agent_response:
            return agent_response["output"]
        else:
            return "Error: No 'output' field found in the agent response."
    except Exception as e:
        return f"Error while extracting output: {str(e)}"

# Tool 1
# Function to execute the two fetching, processing and indexing function
def fetch_and_index_tool(url: str) -> str:
    """
    When given a url,fetches a YouTube transcript, processes it, and indexes it in Pinecone.
    """
    reset_index()
    transcript_data = get_transcript_data_from_url(url)
    process_and_index_transcript(transcript_data)
    return "Transcript successfully fetched, processed, and indexed."

# Tool 2
# Tool to query the vectorstore
def process_user_question(user_question):
    '''
    Given a question as a string, embeds the question and perform similarity search on the Vecstrore to retrieve relevant data.
    '''
    search_results = vectorstore.similarity_search(
    user_question,  # our search query
    k=3
    )
    return search_results

# Definition of the toolbox
tools = [
    Tool(
        name="GetAndIndexTranscript",
        func=fetch_and_index_tool,
        description="Use this tool when given a YouTube video URL to fetch the transcript and indexing it in Pinecone. Input: a YouTube video URL."
    ),
    Tool(
        name="GetRelevantData",
        func=process_user_question,
        description="Use this tool to get relevant data from the transcripts to answer a question. Question must be a string. Input: a query"
    )
]   