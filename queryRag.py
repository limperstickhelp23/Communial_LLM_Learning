from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from seed_db import getEmbedFn

CHROMA_DB = "vec_embeds"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following retrieved context to answer the user's question.
If the answer is not contained in the context, say "I don't know based on the provided information."

Context:
{context}

---
Answer this question based on the context above: {question}
"""


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE.strip(),
    )


def query_rag() -> None:
    embed_fn = getEmbedFn()
    vector_store = Chroma(persist_directory=CHROMA_DB, embedding_function=embed_fn)
    model = OllamaLLM(model="llama3.2:1b")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = build_prompt()

    try:
        while True:
            query = input("🤖 Enter your question: ").strip()
            if not query:
                continue

            results = retriever.invoke(query)
            if not results:
                print("I don't have any matching context for that question yet.")
                continue

            context = "\n---\n".join(doc.page_content for doc in results)
            final_prompt = prompt.format(context=context, question=query)

            response = model.invoke(final_prompt)

            sources = [doc.metadata.get("id") for doc in results]
            formatted_response = f"Response: {response}\n\nSources: {sources}"
            print(formatted_response)
    except KeyboardInterrupt:
        print("\n🦙🦙  Exiting... Thanks for the convo!")


if __name__ == "__main__":
    query_rag()
