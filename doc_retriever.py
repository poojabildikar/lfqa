## Retrieves best matching documents from the document store

from haystack.document_stores.opensearch import OpenSearchDocumentStore
from haystack.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack import Pipeline

from docstore import get_embedding_model

document_store = OpenSearchDocumentStore(
    host="localhost",
    port="9200",
    username="admin",
    password="Portex@5326",
    index="document",
    scheme="https",
    verify_certs=False
)

embedding_model = get_embedding_model()
text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
retriever = OpenSearchEmbeddingRetriever(document_store=document_store)

def get_retriever():
    return OpenSearchEmbeddingRetriever(document_store=document_store)

def get_text_embedder():
    return text_embedder

def retrieve(query):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model))
    query_pipeline.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    result = query_pipeline.run({"text_embedder": {"text": query}})
    result_doc = result['retriever']['documents'][0] 
    print(result_doc)

    return result_doc

retrieve('')