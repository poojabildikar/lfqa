from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from haystack import Document

import os

def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
        doc = Document(content=text, content_type="text")
    return doc


#docs = []
filename = '../data/in_english/Economists.txt'
doc = read_text_file(filename)
#docs.append(doc)


# Create and fill Document Store with document embeddings
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run([doc])
document_store = InMemoryDocumentStore()
document_store.write_documents(docs_with_embeddings["documents"])

#Get Query embedding
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

#Initialize retriever
retriever = InMemoryEmbeddingRetriever(document_store)

#Test retriever
# query="Eligibility for Production Designer"
# text_embedder.warm_up()
# query_embedding = text_embedder.run(query)["embedding"]

# result = retriever.run(query_embedding=query_embedding)
# print(result["documents"])

template = """
Answer the question based only on given content.\n

{% for document in documents %}
    {{ document.content }}
{% endfor %}
\n
Question: {{ question }}?\n
Answer:
"""

#Initialize Generator
generator = LlamaCppGenerator(model="./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_ctx=2048,
    n_batch=128,
    generation_kwargs={"max_tokens": 1024, "temperature": 0.1},
)

pipe = Pipeline()

pipe.add_component("text_embedder", text_embedder)
pipe.add_component("retriever", retriever)
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("text_embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

question = "What are the career options for an Economist in India?"
response = pipe.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(question)
#print(response)
print(response["llm"]["replies"][0])