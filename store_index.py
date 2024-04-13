from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Create or connect to an existing Pinecone index
pc.create_index(
    name=index_name,
    dimension=384, 
    metric='cosine',  
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
index = pc.Index(index_name)

# Load PDF data, split text, and generate embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], index_name=index_name, embedding=embeddings)

# Now you can perform queries against your index
