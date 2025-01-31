##IMPORTS
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import time
import tiktoken
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Loading API's
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Initializing langsmith
os.environ['LANGSMITH_TRACING']='true'
os.environ['LANGSMITH_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGSMITH_API_KEY']=LANGSMITH_API_KEY
os.environ['LANGSMITH_PROJECT']="IH_tracing"

# Initializing tokenizer for pinecode indexing
tiktoken.encoding_for_model('gpt-4o')
tokenizer = tiktoken.get_encoding('o200k_base')

##INITIALIZING PINECONE INDEX

pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

index_name = 'YoutuBot_treasure_box'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)

##EMBEDING MODEL

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

#This import is performed here beacause it conflicts with another Pinecone import
from langchain.vectorstores import Pinecone

text_field = "chunk_text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)