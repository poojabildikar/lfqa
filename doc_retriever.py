## Retrieves best matching documents from the document store

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch.embedding_retriever import OpenSearchEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline

from docstore import get_embedding_model

document_store = OpenSearchDocumentStore(
    host="localhost",
    port="9200",
    http_auth=("admin","Portex@5326"),
    index="initial",
    scheme="https",
    verify_certs=False
)

embedding_model = get_embedding_model()
text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
retriever = OpenSearchEmbeddingRetriever(document_store=document_store, top_k=1)

def get_retriever():
    return retriever

def get_text_embedder():
    return text_embedder

def retrieve(query):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    result = query_pipeline.run({"text_embedder": {"text": query}})
    result_doc = result['retriever']['documents'][0] 
    print("\nSelected document:\n")
    print(result_doc)

    return result_doc

#retrieve('What is Career Margadarshak?')