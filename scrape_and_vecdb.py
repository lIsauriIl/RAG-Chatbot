import bs4
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Scrape and dump info from a webpage of our choice

# Loading parsed webpage
# Define the HTML classes to target specific content on the webpage
classes = ['ch bg fy fz ga gb']

# Initialize the WebBaseLoader with the target webpage and BeautifulSoup parsing options
webloader = WebBaseLoader(
    web_paths=("https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=classes))
)

# Load a PDF document
loader = PyPDFLoader('bert.pdf')
# Extract the first page of the PDF document
docs = [loader.load()[0]]
print(docs)

# Splitting parsed webpage into chunks
# Initialize the text splitter with the desired chunk size and overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
# Split the loaded documents into chunks
chunks = splitter.split_documents(docs)

# Create a Chroma vector database from the document chunks using OllamaEmbeddings
vectordb = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model='mxbai-embed-large'), 
    persist_directory='./vectordb'
)
