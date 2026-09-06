from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore



def process_documnt(document_id:str,file_path:str):
    print(f"Processing document: {document_id}")

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()


        if page_text:
            text += page_text + "\n"

    print(f"Pages : {len(reader.pages)}")

    print(f"Character : {len(text)}")


    # Later:
    #
    # text
    #   ↓
    # chunking
    #   ↓
    # embeddings
    #   ↓
    # vector database

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
    )

    chunks = text_splitter.create_documents([text])


    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag-demo"
    )

    print(f"Document {document_id} processed successfully")

    return {
        "document_id": document_id,
        "status": "completed"
    }





