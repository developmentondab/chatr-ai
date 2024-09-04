pip install google-auth google-auth-oauthlib google-auth-httplib2
pip install google-api-python-client
pip install PyPDF2
pip install faiss-cpu/ faiss-gpu based on python version and OperatingSystem
```
**Note: Enable Google Drive API to get the files.
```
Steps to Generate credentials.json file
1. In the Google Cloud console, go to Menu menu > APIs & Services > Credentials.
   Go to Credentials
2. Click Create Credentials > OAuth client ID.
3. Click Application type > Desktop app.
4. In the Name field, type a name for the credential. This name is only shown in the Google Cloud console.
5. Click Create. The OAuth client created screen appears, showing your new Client ID and Client secret.
6. Click OK. The newly created credential appears under OAuth 2.0 Client IDs.
7. Save the downloaded JSON file as credentials.json, and move the file to your working directory.
