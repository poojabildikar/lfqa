## Docstore - Opensearch
## Semantic matching for document search using SentenceTransformersDocumentEmbedder

from haystack.document_stores.opensearch import OpenSearchDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

import os
import logging

DATA_DIRECTORY = 'data\in-english'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

#Utility function to create document objects from a directory path 
def read_text_files_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Filter for text files
            file_path = os.path.join(directory_path, filename)
            logging.info(f"Reading file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                doc = Document(content=text, content_type="text")
                documents.append(doc)
    return documents

def get_embedding_model():
    return embedding_model

documents = read_text_files_from_directory(DATA_DIRECTORY)

## Generate embeddings for documents
#TODO Figure out the right model to use
#model = "sentence-transformers/all-mpnet-base-v2"
embedding_model="sentence-transformers/all-MiniLM-L6-v2"

document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)  
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store = OpenSearchDocumentStore(
    host="localhost",
    port="9200",
    username="admin",
    password="Portex@5326",
    index="document",
    scheme="https",
    verify_certs=False
)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.SKIP)
print('Number of documents added: ', document_store.count_documents())