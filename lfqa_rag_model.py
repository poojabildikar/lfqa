from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack import Document

from doc_retriever import get_text_embedder, get_retriever

MODEL = "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

def answer(question):
    template = """
    Answer the question based only on given content.\n

    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    \n
    Question: {{ question }}?\n
    Answer:
    """

    text_embedder = get_text_embedder()
    retriever = get_retriever()

    #Initialize Generator
    generator = LlamaCppGenerator(model=MODEL,
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

    response = pipe.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

    print('Question: ', question + '\n')
    print('Answer: ', response["llm"]["replies"][0])

#answer('What is Career Margadarshak?')