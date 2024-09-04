import os
import io
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from PyPDF2 import PdfReader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain

# Scopes for the Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Authenticate and create the Drive API service
def authenticate():
    """Authenticate and create a Google Drive API service."""
    creds = None
    token_file = 'token.pickle'

    # Load existing credentials
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    credentialsFolder = os.path.join(os.getcwd(), "credentials")
    credentialsPath = os.path.join(credentialsFolder, 'credentialsnithin.json')

    # Refresh or obtain new credentials if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentialsPath, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    # Create the Drive API service
    service = build('drive', 'v3', credentials=creds)
    return service

def download_file(service, file_id, file_name):
    """Download a file from Google Drive."""

    # Ensure the folder path exists
    folder_path = 'gdrive'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    print(f"File downloaded: {file_name}")

# List and download files from the specified folder
def list_files_in_folder(service, folder_id):
    """List files in a specific folder in Google Drive and download them."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=10,
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No files found in this folder.')
        return []

    print(f'Files in folder (ID: {folder_id}):')
    downloaded_files = []
    for item in items:
        print(f'{item["name"]} ({item["id"]})')
        file_name = item['name']
        download_file(service, item['id'], file_name)
        downloaded_files.append(file_name)

    return downloaded_files


def get_pdf_data(file_path, num_pages=10):
    file_path = os.path.join('gdrive', file_path)
    reader = PdfReader(file_path)
    full_doc_text = ""
    for page in range(len(reader.pages)):
        current_page = reader.pages[page]
        text = current_page.extract_text()
        full_doc_text += text

    return Document(
        page_content=full_doc_text,
        metadata={"source": file_path}
    )

# Create and save the search index from downloaded files
def create_search_index(docs):
    """Create a search index from the documents and save it."""
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

    for doc in docs:
        source = get_pdf_data(doc)
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    if not source_chunks:
        print("No Docs")
        return False

    faiss_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    # Save the FAISS index using FAISS methods
    faiss_index.save_local("knowledge_bases/gdrive")
    # save_faiss_index(faiss_index.index, "search_index.faiss")    
    print("Index saved to knowledge_bases/gdrive")


def answer_question_with_chain(question):
    """Answer a question using the FAISS index and QA chain."""
    
    loaded_index = FAISS.load_local("knowledge_bases/gdrive", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0), #OpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=loaded_index.as_retriever(search_type="similarity", search_kwargs={"k":6}),
    )

    result = qa(question)
    
    return result

