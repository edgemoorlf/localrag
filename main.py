# 1. Split data into chunks
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator = "\n\n", 
    chunk_size = 7500, 
    chunk_overlap = 100)

from langchain_community.document_loaders import WebBaseLoader
urls = [
    "https://llama.meta.com/llama3/",
    "https://ai.meta.com/blog/meta-llama-3/",
    "https://github.com/meta-llama/llama3"
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
doc_splits = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="./chroma_db"    
)
retriever = vectorstore.as_retriever()

# 3. RAG
rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

from langchain_community.chat_models import ChatOllama
model_local = ChatOllama(model="llama3")
from langchain_core.prompts import ChatPromptTemplate
rag_prompt = ChatPromptTemplate.from_template(rag_template)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)
print(rag_chain.invoke("What is llama3?"))

