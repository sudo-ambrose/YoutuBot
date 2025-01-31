# YoutuBot - YouTube Video Query Agent

## Project Overview
YoutuBot is an **agentic system** designed to **retrieve and answer queries about YouTube videos**. It processes **one video at a time**, clearing its database with each new URL provided. The system is powered by `OpenAI's GPT-4o` ensuring high-quality query understanding and response generation.

## Architecture
YoutuBot is built using the **LangChain** library and consists of:
- **Chat Completion LLM** (GPT-4o)
- **Pinecone VectorStore** for efficient document retrieval
- **Simple retriever** (Cosine Similarity Search)
- **Two tools** for data processing and querying
- **OpenAI Functions type Agent** with a fully custom system prompt

## How It Works
### **Data Fetching**
- The Pinecone vector store **stores transcripts of a single YouTube video** at a time, using the `Youtube_Transcript_API` library.
- Each transcript is split into **400-token chunks** (with a 20-token overlap to prevent information loss).
- Text chunks are embedded using **OpenAI's `text-embedding-ada-002` model**.
- The indexed metadata enables fast **cosine similarity search** for relevant content.

### **Agent Behavior**
The **OpenAI Functions type Agent** operates purely based on a **custom system prompt**, structured around four key aspects:
1. **Agent’s Role & Abilities**
2. **Rules & Workflow**
3. **Answering Questions**
4. **Resetting for a New Video**

To maintain context in conversations, a **Conversational Buffer Memory (7 past interactions)** is used.

### **Tools & Processing Flow**
1. **Video Processing Tool**
   - Clears the vector store
   - Fetches the transcript & metadata
   - Splits and indexes transcript chunks into the vector store
2. **Query Processing Tool**
   - Performs **Cosine Similarity Search** in the vector store
   - Returns the **top 3 relevant chunks**

## Improvements & Evaluation
### **Key Optimizations**
- Initially, the agent relied on **LangChain’s agent base system prompt**, but a **fully custom prompt** proved to be **more effective**, allowing precise control.
- The retriever was **originally a QA chain** returning only LLM-generated answers. Switching to **direct document retrieval** provided **more accurate responses**.
- Instead of standard **question-based similarity search**, the agent now **uses keywords and adapted statements** to retrieve more relevant chunks.

### **Evaluation Process**
- Conducted via **LangSmith** & manual **chat-based testing**.
- The agent correctly adheres to its **primary role**: **describing video content** rather than providing general knowledge.
- **ROUGE/BLEU metrics** were **not** used, as multiple valid phrasings exist for responses. However, these metrics would be necessary for real-world deployment.

## Deployment & Usage
### **Interface**
- A **Gradio chatbot interface** created for interaction.
- Hosted **locally**.

### **Usage Example**
1. **Provide a YouTube video link** (The agent extracts the transcript & indexes it).
2. **Ask a question about the video** (The agent retrieves the most relevant data).
3. **Receive responses based on accurate transcript retrieval.**

## Future Considerations
- **Error handling & fallback mechanisms** for unavailable transcripts.
- **Support for multi-video storage & retrieval**.
- **Deployment to a cloud-based service** for public accessibility.

---

## Files
- **main.py** : containing gradio code
- **agent.py** : containing agent related code
- **functions.py** : containing all the functions
- **variables.py** : containing initialized variables
- **requirements.txt** : necessary libraries
- **YoutuBot.pptx** : presentation

