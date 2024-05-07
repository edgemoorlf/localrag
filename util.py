from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings


def prepareRag(lists) -> VectorStoreRetriever:

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n\n", 
        chunk_size = 7500, 
        chunk_overlap = 100)

    docs = [WebBaseLoader(url).load() for url in lists]
    docs_list = [item for sublist in docs for item in sublist]
    doc_splits = text_splitter.split_documents(docs_list)

    # 2. Convert documents to Embeddings and store them
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    return retriever
