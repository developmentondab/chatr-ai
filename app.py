import os
import pathlib
import openai
import shutil
from dotenv import load_dotenv
import re
import secrets
import random, string
import json
import datetime
import random
import string

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import (
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
    JSONLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException, Depends, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

import database, url_slug
import logging

from pydantic import BaseModel
import mysql_db
from auth import (
    get_hashed_password,
    create_access_token,
    create_refresh_token,
    verify_password
)
from auth_bearer import JWTBearer
import gdrive_search


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".json": (JSONLoader, {"jq_schema":"."}),
    ".xlsx": (UnstructuredExcelLoader, {"mode":"elements", "encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader, {"mode":"elements", "encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

app = FastAPI()

# Set the maximum upload size to 10 megabytes
app = FastAPI(upload_max_size="50MB")
# app.config.max_request_size = 50 * 1024 * 1024
app.max_request_size = 50 * 1024 * 1024

# Pydantic model to define the schema of the data
class Item(BaseModel):
    shared_content: str
    collection: str

class Subscription(BaseModel):
    email: str
    sub_id: str
    name: str = None
    start_date: int
    end_date: int
    invoice_url: str
    status: int = 1    

router = APIRouter(prefix='/chat')

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

def generate_random_string(length):
    random_string = ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(length))
    random_string += '=='
    return random_string

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
        if file_path.endswith(".json"):
            # Read JSON data from the file
            with open(file_path, 'r') as file:
                json_data = json.load(file)

            # Convert JSON to text
            pages = json.dumps(json_data, indent=2)
            merged_docs.extend([Document(page_content= pages, metadata= {"source": file_path})])
        else:
            merged_docs.extend(load_single_document(file_path))

    # return [load_single_document(file_path) for file_path in source_dir]
    return merged_docs

@router.post("/api/uploadfile/")
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

@router.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile], user_id: str = Depends(JWTBearer()), kb_name: Optional[str] = None):
    upload_dir = pathlib.Path(os.getcwd(), "data")

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    if(user['first_name'] == 'Guest' or 'dummy' in user['email']):
        can_proceed = check_daily_quota('files', files, user_id)
        if can_proceed['status'] == False:
            limit = ""
            if 'limit' in can_proceed:
                limit = "end"
            return JSONResponse(content={"message":can_proceed['message'], "status": False, "limit": limit}, status_code=422)
            
    saved_files = []
    collection_name = '' if (kb_name is None or kb_name == "") else kb_name.replace(" ", "_")
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
    
    kb_name = collection_name if (kb_name is None or kb_name == "") else kb_name
    database.update({collection_name:{"file_paths":saved_files, "file_names":file_names, "kb_name":kb_name, "data_type":"files"}})
    data = {"collection_name": collection_name, "user_id": user_id, "file_paths":saved_files, "file_names":file_names, "kb_name":kb_name, "data_type": "files"}
    mysql_db.add_collection(data)

    # do something with the file
    return {"filenames": saved_files}

@router.get("/chat_new")
def read_root(query: str, collection: str, share_user_id: Optional[str] = None, user_id: str = Depends(JWTBearer())):
    user_id = user_id if share_user_id is None else share_user_id
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.load_local("knowledge_bases/"+collection, embeddings)

    # template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    # just say that you don't know, don't try to make up an answer.
    #  and provide three follow-on questions to that answer, two follow-on questions related to document and Start your response with "follow-ques:" for each question
    template = """You are tailored to provide information based on a wide range of topics, exclusively drawing from the content of uploaded files, documents, web links, and YouTube video transcripts. It will focus on speaking directly from these data sources, ensuring accuracy and relevance. Info Navigator will not give opinions or speculations and will refrain from discussing topics outside the provided materials. It adopts a clear and concise communication style, with a slight infusion of friendliness to make interactions engaging.

    {context}

    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(
        llm=llm, #OpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=loaded_index.as_retriever(search_type="similarity", search_kwargs={"k":6}),
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

@router.get("/collections")
def get_collections(key: Optional[str] = None, user_id: str = Depends(JWTBearer())):
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=404)

    if key is not None:
        collections = mysql_db.get_collection_bykey(user_id, key)
    else:
        collections = mysql_db.get_collection(user_id)

    # print(database.collections())
    if(not collections):
        return JSONResponse(content={"message":"No records found", "status": False}, status_code=404)
    
    return {"collections": collections}

@router.get("/remove_collection")
def remove_collections(collection: str, user_id: str = Depends(JWTBearer())):
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=404)
    
    collections = mysql_db.get_collection_bykey(user_id, collection)
    if(not collections):
        return JSONResponse(content={"message":"No records found", "status": False}, status_code=404)

    for file_path in collections['file_paths']:
        os.remove(file_path)
    
    if os.path.exists("knowledge_bases/"+collection):    
        shutil.rmtree("knowledge_bases/"+collection)
    # database.remove(collection)
    mysql_db.remove_collection(user_id, collection)

    return {"status": "Removed the collection", "collections":database.collections() }

@router.post("/add_url/")
async def web_loader(web_url: list, kb_name: Optional[str]=None, url_type: Optional[str]=None, user_id: str = Depends(JWTBearer())):
    for utube in web_url:
        web_url = utube.split(',')
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    if(user['first_name'] == 'Guest' or 'dummy' in user['email']):
        can_proceed = check_daily_quota('urls', web_url, user_id)
        if can_proceed['status'] == False:
            limit = ""
            if 'limit' in can_proceed:
                limit = "end"
            return JSONResponse(content={"message":can_proceed['message'], "status": False, "limit": limit}, status_code=422)
    
    logger.info(f"---Kb_Name: {kb_name}")
    collection = '' if (kb_name is None or kb_name == "") else kb_name.replace(" ", "_")
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
        data = []
        for weburl in web_url:
            logger.info(f"---Tera URL: {weburl}")
            loader = WebBaseLoader(weburl)
            data.extend(loader.load())
        # loader = WebBaseLoader(web_url)
        # data = loader.load()
    logger.info(f"---teraData: {data}")
    merged_content = "\n".join(doc.page_content.replace("\n", "") for doc in data)
    logger.info(f"---teraData merged_content: {merged_content}")
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
        
    kb_name = collection if (kb_name is None or kb_name == "") else kb_name    
    database.update({collection:{"file_paths":[], "file_names":urls_data, "kb_name":kb_name, "data_type":"urls"}})
    data = {"collection_name": collection, "user_id": user_id, "file_paths":[], "file_names":urls_data, "kb_name":kb_name, "data_type": "urls"}
    mysql_db.add_collection(data)

    return {"status":"success", "data": "all OK"}

@router.get("/file/{file_name}")
async def get_file(file_name: str):
    file_path = pathlib.Path(os.getcwd(), "data", file_name)
    if(os.path.exists(file_path)):
        return FileResponse(file_path)
    else:
        return HTTPException(status_code=404)

def get_questions(llm, context, question, answer):
    template = """Use this context, question and answer to provide three follow-on questions. Start your response with "follow-ques:" for each question. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    Answer: {answer}

    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question", "answer"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"context":context,"question":question,"answer":answer})
    return resp

@router.get("/file_summary")
def file_summary(collection: str, request: Request, user_id: str = Depends(JWTBearer())):
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    collection_path = "knowledge_bases/"+collection
    embeddings = OpenAIEmbeddings()
    loaded_index = FAISS.load_local(collection_path, embeddings)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    #  Summary the docs
    data = list(loaded_index.docstore._dict.values())
    if (len(data[0:1][0].page_content) <= 500):
        merge_len = 19
    else:
        merge_len = 9
    
    merged_parts = []
    
    no_of_parts = int((len(data)/merge_len)+0.5)
    present_stage = 0
    for _ in range(no_of_parts):
        merged_page_content = ' '.join(item.page_content for item in data[present_stage:(present_stage+merge_len)])
        merged_parts.extend([Document(page_content= merged_page_content, metadata= data[present_stage:(present_stage+merge_len)][0].metadata)])
        present_stage = present_stage+merge_len+1

    map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )
    
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
    # outpt = summary_chain.run(list(loaded_index.docstore._dict.values())[0:50])
    outpt = summary_chain.run(merged_parts)

    # collections = database.get_collection(collection)
    collections = mysql_db.get_collection_bykey(user_id, collection)
    
    if(not collections):
        return JSONResponse(content={"message":"No records found", "status": False}, status_code=404)
    
    return {"status": "Success", "summary": outpt, "collections": collections, "base_url": "https://dochat.terasoftware.com/file/"}

@router.get("/update_db")
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

@router.post("/register/")
async def register(username: str, email: str, password: str, first_name: str, last_name: str = None):
    data = {
        "username": username,
        "email": email,
        "password": get_hashed_password(password),
        "first_name": first_name,
        "last_name": last_name
    }
    user = mysql_db.check_email(email)
    if(user):
        return JSONResponse(content={"message":"Email already exists.", "status": False}, status_code=400)

    resp = mysql_db.add_user(data)

    if(resp):
        user = mysql_db.get_user(resp)

        # Adding 14 days free Trail when a new user registers.
        sub_data = {
            "email": email,
            "invoice_url": "",
            "start_date": datetime.datetime.now().timestamp(),
            "end_date": datetime.datetime.now().timestamp() + (14 * 24 * 60 * 60),
            "sub_id": datetime.datetime.now().timestamp(),
            "name": "Free Trail",
            "status": 1
        }
        respp = mysql_db.add_subscription_data(sub_data, user['user_id'])
        logger.info(f"Add User SubScription: {respp} ")

        access_token = create_access_token(user['username'], user['user_id'])
        refresh_token = create_refresh_token(user['username'], user['user_id'])
        mysql_db.add_tokens(user['user_id'], access_token, refresh_token)
        return {"message":"User created successfully", "status":True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "name":user['first_name'] + " " +'' if user['last_name'] is None else str(user['last_name']),
            "email":user['email'],
            "username":user['username']}
    else:
        return JSONResponse(content={"message":"Error while creating the user.", "status": False}, status_code=400)

@router.post("/login/")
async def login(username: str, password: str):
    data = {
        "username": username,
        "password": password
    }
    user = mysql_db.check_email(username)
    if(not user):
        return JSONResponse(content={"message":"Incorrect username or password", "status": False}, status_code=400)
    
    if not verify_password(password, user['password']):
        return JSONResponse(content={"message":"Incorrect username or password", "status": False}, status_code=400)
    
    if(user):

        access_token = create_access_token(user['username'], user['user_id'])
        refresh_token = create_refresh_token(user['username'], user['user_id'])
        mysql_db.add_tokens(user['user_id'], access_token, refresh_token)
        return {"message":"Login successfull", "status":True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "name":user['first_name'] + " " +'' if user['last_name'] is None else str(user['last_name']),
            "email":user['email'],
            "username":user['username']
        }
    else:
        return JSONResponse(content={"message":"Invalid Credentails", "status": False}, status_code=400)

@router.get("/profile")
def profile(user_id: str = Depends(JWTBearer())):
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    return {"message":"User Details", "status":True, "data": user}

@router.post('/test-token', summary="Test if the access token is valid")
async def test_token(user_id: str = Depends(JWTBearer())):
    return user_id

@router.post("/logout/")
async def logout(user_id: str = Depends(JWTBearer())):
    mysql_db.revoke_token(user_id)
    return JSONResponse(content={"message":"User logged out successfully.", "status": True}, status_code=200)

@router.post("/update_profile")
async def update_profile(username: str, email: str, first_name: str, last_name: str = None, user_id: str = Depends(JWTBearer())):
    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    data = {
        "username": username,
        "email": email,
        "first_name": first_name,
        "last_name": last_name
    }

    resp = mysql_db.update_user(data, user['user_id'])
    if resp:
        return JSONResponse(content={"message":"User updated successfully.", "status": False}, status_code=200)
    else:
        return JSONResponse(content={"message":"Updation failed.", "status": False}, status_code=400)

@router.post("/google_login/")
async def google_login(google_id: str, email: str, given_name: str = None, family_name: str = None):    
    password = secrets.token_urlsafe(10)
    data = {
        "google_id": google_id,
        "first_name": given_name,
        "last_name": family_name,
        "email": email,
        "password": password,
        "username": given_name
    }
    user = mysql_db.check_email(email)
    resp = None
    if(not user):
        resp = mysql_db.add_user(data)
    
    if(resp):
        user = mysql_db.get_user(resp)

    if(user):
        access_token = create_access_token(user['username'], user['user_id'])
        refresh_token = create_refresh_token(user['username'], user['user_id'])
        mysql_db.add_tokens(user['user_id'], access_token, refresh_token)
        return {"message":"Login successfull", "status":True,
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    else:
        return JSONResponse(content={"message":"Invalid Credentails", "status": False}, status_code=400)

@router.post("/get_shared_content/")
async def get_shared_content(share_id: str, get_token: str = None):   
    if get_token is not None:  
        resp = mysql_db.get_share_user_data(share_id)        
    else:
        resp = mysql_db.get_share_data(share_id)
    
    if(not resp):
        return JSONResponse(content={"message":"No Records Found!", "status": False}, status_code=400)

    if get_token is not None:
        user = mysql_db.get_user(resp[0]['shared_user_id'])
        if(user):
            access_token = create_access_token(user['username'], user['user_id'])
            refresh_token = create_refresh_token(user['username'], user['user_id'])
            mysql_db.add_tokens(user['user_id'], access_token, refresh_token)
            return {"message":"Shared user token successfull", "status":True,
                "access_token": access_token,
                "refresh_token": refresh_token
            }
        else:
            return JSONResponse(content={"message":"User Not Found!", "status": False}, status_code=400)

    return {"status": True, "data": resp[0]}

@router.post("/add_shared_content/")
async def add_shared_content(item: Item, user_id: str = Depends(JWTBearer())): 
    share_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    check_data = mysql_db.check_share_data(user_id, item.collection) 
    if(not check_data):
        resp = mysql_db.add_share_data(share_id, user_id, item.collection, item.shared_content)
    else:        
        share_id = check_data[0]['share_id']
        resp = mysql_db.update_share_data(check_data[0]['id'], item.shared_content)  
    
    if(not resp):
        return JSONResponse(content={"message":"Error Adding Content!", "status": False}, status_code=400)

    return {"status": True, "data": share_id}

@router.post("/add_subscription/")
async def add_subscription(subscription: Subscription): 
    try:
        check_data = mysql_db.check_email(subscription.email)
        if(not check_data):
            return JSONResponse(content={"message":"No user with given email!", "status": False}, status_code=400)

        resp = mysql_db.add_subscription_data(subscription.dict(), check_data['user_id'])
            
        if(not resp):
            return JSONResponse(content={"message":"Error Adding Subscription!", "status": False}, status_code=400)

        return {"status": True, "data": resp}
    except:
        return JSONResponse(content={"message":"Something Went Wrong!", "status": False}, status_code=400)

@router.get("/get_subscription")
def get_subscription(user_id: str = Depends(JWTBearer())): 

    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    if(user['first_name'] == 'Guest' or 'dummy' in user['email']):
        return {
            "status": True, 
            "data": {
                "user_id": user['user_id'], 
                "sub_id": "sub_id_dummy", 
                "name": "Free Trail",
                "start_date": "2024-04-01 13:10:03",  
                "end_date": "2224-04-01 13:10:03", 
                "invoice_url": "",
                "status": 1
            }
        }

    resp = mysql_db.get_subscription(user_id)
        
    if(not resp):
        return JSONResponse(content={"message":"Error Getting Subscription!", "status": False}, status_code=400)

    return {"status": True, "data": resp[0]}

@router.post("/multi_upload/")
async def multi_upload(web_url: list = None, files:List[UploadFile] = File(None), user_id: str = Depends(JWTBearer()), kb_name: Optional[str] = None):
    
    if web_url is None and files is None:
        return JSONResponse(content={"message":"At least one of parameter web_url or files should be provided.", "status": False}, status_code=400)

    if not any(web_url) and files is None:
        return JSONResponse(content={"message":"At least one of parameter web_url or files should be provided.", "status": False}, status_code=400)

    # return {"web_url": web_url, "user_id": user_id, "kb_name": kb_name}
    upload_dir = pathlib.Path(os.getcwd(), "data")

    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    user = mysql_db.get_user(user_id)
    if(not user):
        return JSONResponse(content={"message":"User not found.", "status": False}, status_code=400)

    saved_files = []
    collection_name = '' if (kb_name is None or kb_name == "") else kb_name.replace(" ", "_")
    file_names = []
    data = []
    url_type = ""

    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        saved_files.append(file_path)
        file_names.append(file.filename)
        if(collection_name == ''):
            collection_name = file.filename.replace(" ", "_")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
    data = load_documents(saved_files) 

    # Urls data into file and vector
    print(web_url)
    logger.info(f"---web_url: {web_url}")
    for utube in web_url:
        web_url = utube.split(',')
    print(web_url)
    logger.info(f"---web_url: {web_url}")
        
    if(collection_name == ''):
        collection_name, urls = url_slug.get_slug_from_url(web_url)
    url_data = []
    for utube in web_url:
        if "youtube" in utube or "youtu.be" in utube:
            url_type = "youtube"
            loaderr = YoutubeLoader.from_youtube_url(utube, add_video_info=True)
            url_data.extend(loaderr.load())
        else:
            loadeer = WebBaseLoader(utube)
            url_data.extend(loadeer.load())

    data.extend(url_data)
    merged_content = "\n".join(doc.page_content.replace("\n", "") for doc in url_data)    
    output_file = pathlib.Path(os.getcwd(), "data", collection_name+".txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(merged_content)  
    # saved_files.append(output_file)

    # Url Code
    for url in web_url:
        file_names.extend(url.split(','))

    if any(web_url):
        file_names.append(collection_name+".txt")

    # Creating the chunk based on the content
    chunk_size = 500
    chunk_overlap = 50
    if url_type == 'youtube':
        chunk_size = 1000
        chunk_overlap = 100

    # Split the large content to small chunks and create vector for them.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()    
    loaded_index = FAISS.from_documents(texts, embeddings)
    loaded_index.save_local("knowledge_bases/"+collection_name)
    # Save the index to a file ... Later, when you want to use the index ...
    
    
    kb_name = collection_name if (kb_name is None or kb_name == "") else kb_name
    # database.update({collection_name:{"file_paths":saved_files, "file_names":file_names, "kb_name":kb_name, "data_type":"multiple"}})
    data = {"collection_name": collection_name, "user_id": user_id, "file_paths":saved_files, "file_names":file_names, "kb_name":kb_name, "data_type": "multiple"}
    mysql_db.add_collection(data)

    return {"filenames": saved_files}


@router.post("/dummy_login/")
async def dummy_login(ip_address: str, cookie_id: str = None):

    if(ip_address is None):
        return JSONResponse(content={"message":"Ip address cannot be empty!", "status": False}, status_code=400)
    ip_address = ip_address.replace(".", "_")

    data = {
        "username": "Guest",
        "email": "dummy"+ip_address+"@gmail.com",
        "password": get_hashed_password("abc"+ip_address),
        "first_name": "Guest",
        "last_name": "user",
        "cookie_id": generate_random_string(32)
    }
    user = mysql_db.check_cookieid_email(cookie_id, data['email'])
    if(not user):
        resp = mysql_db.add_user(data)
        if(resp):
            user = mysql_db.get_user(resp)
        else:
            return JSONResponse(content={"message":"Error while creating the dummy user.", "status": False}, status_code=400)
    
    if(user):

        access_token = create_access_token(user['username'], user['user_id'])
        refresh_token = create_refresh_token(user['username'], user['user_id'])
        mysql_db.add_tokens(user['user_id'], access_token, refresh_token)
        return {"message":"Guest user successfull", "status":True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "name":user['first_name'] + " " +'' if user['last_name'] is None else str(user['last_name']),
            "email":user['email'],
            "username":user['username'],
            "cookie_id": user['cookie_id']
        }
    else:
        return JSONResponse(content={"message":"Invalid Details!", "status": False}, status_code=400)

def check_daily_quota(type: str, files, user_id):

    max_websites_per_day  = 2
    files_per_day = 2
    max_size_per_day  = 5 * 1024 * 1024  # 5 MB

    # Checking in DB if user has already records for today
    urls_count = files_count = 0
    collections = mysql_db.get_collection_byDate(user_id, datetime.date.today())
    if(collections):
        for key, collection in collections.items():
            data_type = collection['data_type']
            if data_type == 'urls':
                file_names_count = len(collection['file_names']) - 1
                if file_names_count > 1:
                    urls_count += file_names_count
                else:
                    urls_count += 1 

            elif data_type == 'files':                       
                file_paths_count = len(collection['file_paths'])
                if file_paths_count > 1:
                    files_count += file_paths_count
                else:
                    files_count += 1

    if urls_count >= 2 and files_count >= 2:
        return { "status": False, "message": "Daily quota exceeded. Please try again tomorrow.", "limit":"end"}
    elif urls_count > 2:
        return { "status": False, "message": "Only 2 links are allowed per day!"}
    elif files_count > 2:
        return { "status": False, "message": "Only 2 pdfs are allowed per day!"}
    else:
        max_websites_per_day = max_websites_per_day - urls_count
        files_per_day = files_per_day - files_count

    if type == 'files':       
        if len(files) > files_per_day:
            return { "status": False, "message": "Only 2 pdfs are allowed per day!"} 

        for file in files:
            _, file_extension = os.path.splitext(file.filename)
            if(file_extension != '.pdf'):
                return { "status": False, "message": "Only .pdfs are accepted!"}
            
            if file.size > max_size_per_day:
                return { "status": False, "message": "please upload files < 5MB only"}
        
        return { "status": True, "message": ""}

    elif type == "urls":
        if len(files) > max_websites_per_day:
            return { "status": False, "message": "Only 2 links are allowed per day!"}

        return { "status": True, "message": ""}

    return { "status": True, "message": ""}

@router.get("/auth_drive")
def auth_drive():
    try:
        folder_id = '<YOUR_DRIVE_FOLDER_ID>'
        # Authenticate and create the Drive API service
        service = gdrive_search.authenticate()

        # List and download files from the specified folder
        downloaded_files = gdrive_search.list_files_in_folder(service, folder_id)
        # downloaded_files = ["Copy of DraftRFPABP08012024.pdf", 'DraftRFPABP08012024.pdf', 'Invoice-A3809B34-0003.pdf']
    
        if downloaded_files:
            gdrive_search.create_search_index(downloaded_files)
            return JSONResponse(content={"message":"Success!", "status": True}, status_code=200)
        else:
            print("No files to index.")
            return JSONResponse(content={"message":"No files to index.", "status": False}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"message":"Something Went Wrong!", "status": False, "data": str(e)}, status_code=400)

@router.post("/answer_doc")
def answer_doc(question: str):
    try:
        answer = gdrive_search.answer_question_with_chain(question)
        return JSONResponse(content={"message":"Success!", "status": True, "data": answer}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message":"Something Went Wrong!", "status": False, "data": str(e)}, status_code=400)

app.include_router(router)    
# query = input("Ask me anything? ")
# print(qa.run(query))
