import os
import pathlib
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional



# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
        # pages = loader.load_and_split()
        # return pages
        # return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir) -> List[Document]:
    # Loads all documents from source documents directory
    # all_files = []
    # for ext in LOADER_MAPPING:
    #     all_files.extend(
    #         glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
    #     )
    merged_docs = []

    for file_path in source_dir:
        merged_docs.extend(load_single_document(file_path))

    # return [load_single_document(file_path) for file_path in source_dir]
    return merged_docs

@app.post("/api/uploadfile/")
async def create_upload_file1(file: UploadFile):
    upload_dir = os.path.join(os.getcwd(), "data")

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    file_path = os.path.join(upload_dir, file.filename)
    print(file_path)
    
    with open(file_path, "wb") as f:
            f.write(await file.read())

    pages = load_single_document(file_path)
    # loader = PyPDFLoader("data/Tamil_Nadu_Tender.pdf")
    # pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()

    loaded_index = FAISS.from_documents(pages, embeddings)
    # Save the index to a file "saved_index.index changed to saved_index"
    # FAISS.write_index(loaded_index.index, "saved_index")
    loaded_index.save_local("saved_index")
    # ... Later, when you want to use the index ...
    
    # do something with the file
    return {"filename": file.filename}

@app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile]):
    upload_dir = pathlib.Path(os.getcwd(), "data")

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    saved_files = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        saved_files.append(file_path)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

    chunk_size = 500
    chunk_overlap = 50
    documents = load_documents(saved_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    
    loaded_index = FAISS.from_documents(texts, embeddings)
    # Save the index to a file "saved_index.index changed to saved_index"
    # FAISS.write_index(loaded_index.index, "saved_index")
    loaded_index.save_local("saved_index")
    # ... Later, when you want to use the index ...
    
    # do something with the file
    return {"filenames": saved_files}

@app.get("/chat")
def read_root(query: str):
    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.load_local("saved_index", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=loaded_index.as_retriever(),
        return_source_documents=True
    )

    return {"status": "Success", "query": qa(query)['result']}

# query = input("Ask me anything? ")
# print(qa.run(query))
