import os
import pathlib
import openai
import shutil
from dotenv import load_dotenv
import re

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
    WebBaseLoader,
    YoutubeLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

import database, url_slug
import logging


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

# Create a logger
logging.basicConfig(filename='api_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def preflight_handler(request: Request):
    if request.method == "OPTIONS":
        return {}
        
app.options("/{path:path}", response_model=None, include_in_schema=False)(preflight_handler)


# Middleware to log API requests and responses
@app.middleware("http")
async def log_api_requests(request: Request, call_next):
    if "chat" in str(request.url) or "add_url" in str(request.url):
        # Log the request
        logger.info(f"Request: {request.method} {request.url}")

    # Call the next middleware
    response = await call_next(request)

    if "chat" in str(request.url) or "add_url" in str(request.url):
        # Log the response    
        logger.info(f"Response: {response.status_code}")
        logger.info("-------------__________________------------------")

    # logger.info(f"Response message: {response_bytes.body.decode('utf-8')}")
    # response.headers["Access-Control-Allow-Origin"] = '*'
    # response.headers["Access-Control-Allow-Credentials"] = 'true'
    # response.headers["Access-Control-Allow-Headers"] = '*'
    return response

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
async def create_upload_file(files: List[UploadFile], kb_name: Optional[str] = None):
    upload_dir = pathlib.Path(os.getcwd(), "data")

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    saved_files = []
    collection_name = '' if kb_name is None else kb_name.replace(" ", "_")
    file_names = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        saved_files.append(file_path)
        file_names.append(file.filename)
        if(collection_name == ''):
            collection_name = file.filename.replace(" ", "_")
        
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
    loaded_index.save_local("knowledge_bases/"+collection_name)
    # ... Later, when you want to use the index ...
    
    database.update({collection_name:{"file_paths":saved_files, "file_names":file_names, "kb_name":kb_name, "data_type":"files"}})

    # do something with the file
    return {"filenames": saved_files}

@app.get("/chat_new")
def read_root(query: str, collection: str):
    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.load_local("knowledge_bases/"+collection, embeddings)

    # template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    # just say that you don't know, don't try to make up an answer.
    #  and provide three follow-on questions to that answer, two follow-on questions related to document and Start your response with "follow-ques:" for each question
    template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(
        llm=llm, #OpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=loaded_index.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    res = qa(query)

    unique_page_numbers = sorted(set(document.metadata.get("page", "") for document in res['source_documents']))
    
    # Merge the page_content values into a single string
    merged_content = ""
    if(res['source_documents']):
        merged_content = "\n".join(doc.page_content for doc in res['source_documents'])
    # logger.info(f"Merged Content: {merged_content}")
    output = get_questions(llm, merged_content, query, res['result'])
    logger.info(f"Merged Content OUTPUT: {output}")

    # new_res = {}
    # one = res['result'].split('\nFollow-on questions:')
    # if len(one) > 1:
    #     two = one[1].split('\nFollow-on questions related to the document:')
        
    #     new_res['answer'] = one[0]
    #     split_sections = re.split(r'\n\d+[.\)]', two[0])
    #     new_res['fquestions'] = [q.strip() for q in split_sections if q.strip()]
        
    #     if len(two) > 1:
    #         split_sections = re.split(r'\n\d+[.\)]', two[1])
    #         new_res['fdquestions'] = [q.strip() for q in split_sections if q.strip()]
    #     else:
    #         new_res['fdquestions'] = []  # No related questions found
    
    new_res = {}
    one = re.split(r'(?<=\n)follow-ques:', output)
    if len(one) > 1:
        new_res['answer'] = res['result']
        one.pop(0)
        new_res['fquestions'] = [q.strip() for q in one if q.strip()]
        new_res['fdquestions'] = []
    else:
        new_res['answer'] = res['result']
        new_res['fquestions'] = []
        new_res['fdquestions'] = []

    # return {"status": "Success", "query": res['result'], "page_numbers": unique_page_numbers}
    # {"status": "Success", "query": new_res['answer'], "page_numbers": unique_page_numbers, "questions":new_res, 'resps':res['result']}
    data = {"status": "Success", "query": new_res['answer'], "page_numbers": unique_page_numbers, "questions":new_res, 'resps':res['result']}
    logger.info(f"Response message: {data}")
    return data

@app.get("/collections")
def get_collections(key: Optional[str] = None):
    
    if key is not None:
        collections = database.get_collection(key)
    else:
        collections = database.collections()
    # print(database.collections())
    collections = dict(reversed(collections.items()))
    return {"collections": collections}

@app.get("/remove_collection")
def remove_collections(collection: str):
    files = database.get_collection(collection)

    for file_path in files['file_paths']:
        os.remove(file_path)
        
    shutil.rmtree("knowledge_bases/"+collection)
    database.remove(collection)

    return {"status": "Removed the collection", "collections":database.collections() }

@app.post("/add_url/")
async def web_loader(web_url: list, kb_name: Optional[str]=None, url_type: Optional[str]=None):
    logger.info(f"---Kb_Name: {kb_name}")
    collection = '' if kb_name is None else kb_name.replace(" ", "_")
    if(collection == ''):
        collection, urls = url_slug.get_slug_from_url(web_url)
    logger.info(f"---collection: {collection}")
    chunk_size = 500
    chunk_overlap = 50
    if(url_type == 'youtube'):
        chunk_size = 1000
        chunk_overlap = 100
        data = []
        for utube in web_url:
            loader = YoutubeLoader.from_youtube_url(utube, add_video_info=True)
            data.extend(loader.load())
    else:
        loader = WebBaseLoader(web_url)
        data = loader.load()
    
    merged_content = "\n".join(doc.page_content.replace("\n", "") for doc in data)
    
    output_file = pathlib.Path(os.getcwd(), "data", collection+".txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(merged_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    all_splits = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.from_documents(all_splits, embeddings)
    loaded_index.save_local("knowledge_bases/"+collection)

    urls_data = [collection+".txt"]
    for url in web_url:
        urls_data.extend(url.split(','))
        
    database.update({collection:{"file_paths":[], "file_names":urls_data, "kb_name":kb_name, "data_type":"urls"}})

    return {"status":"success", "data": "all OK"}

@app.get("/file/{file_name}")
async def get_file(file_name: str):
    file_path = pathlib.Path(os.getcwd(), "data", file_name)
    if(os.path.exists(file_path)):
        return FileResponse(file_path)
    else:
        return HTTPException(status_code=404)

def get_questions(llm, context, question, answer):
    template = """Use this context, question and answer to provide four follow-on questions. Start your response with "follow-ques:" for each question. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    Answer: {answer}

    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question", "answer"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"context":context,"question":question,"answer":answer})
    return resp

@app.get("/file_summary")
def file_summary(collection: str, request: Request):

    collection_path = "knowledge_bases/"+collection
    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.load_local(collection_path, embeddings)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    #  Summary the docs
    summary_chain = load_summarize_chain(llm, chain_type="stuff")
    outpt = summary_chain.run(list(loaded_index.docstore._dict.values()))

    collections = database.get_collection(collection)
    
    return {"status": "Success", "summary": outpt, "collections": collections, "base_url": "https://dochat.terasoftware.com/file/"}

@app.get("/update_db")
def update_db(collection):
    collections = database.get_collection(collection)

    # embeddings = OpenAIEmbeddings()
    # loaded_index = FAISS.load_local("knowledge_bases/"+collection, embeddings)
    # data = list(loaded_index.docstore._dict.values())

    # merged_content = "\n".join(doc.page_content.replace("\n", "") for doc in data)
    
    # output_file = pathlib.Path(os.getcwd(), "data", collection+".txt")
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     file.write(merged_content)

    # file_names = [collection+".txt"]
    # for url in collections['file_names']:
    #     file_names.extend(url.split(','))

    database.update({collection:{"file_paths":collections['file_paths'], "file_names":collections['file_names'], "kb_name":None, "data_type":"files"}})
    return "ok"

# query = input("Ask me anything? ")
# print(qa.run(query))
