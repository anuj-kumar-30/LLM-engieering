# 1. Imports
import os
import glob
from dotenv import load_dotenv
import gradio as gr

# langchain imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Creating db_name, and model variables
model = 'gemini-2.0-flash'
db_name = 'vector_db'

# Read in documents using langchain loaders
# take everything in all the sub folders of our knowledge base
folders = glob.glob('knowledge-base/*')
print(folders)

text_loader_kwargs = {'encoding':'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder) # prints the sub-folder inside knowledge-base 
    # print(doc_type)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs) # creates object class for each subfolder
    # print(loader)
    folder_docs = loader.load() # loads all the content inside the documents
    # print(folder_docs)
    for doc in folder_docs:
        doc.metadata['doc_type']=doc_type
        documents.append(doc)

print(len(documents))        
