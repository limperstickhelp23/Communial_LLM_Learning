from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader, 
    StorageContext,
    VectorStoreIndex, 
    get_response_synthesizer, 
    Settings
)

import chromadb
import argparse
import os

ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_ORANGE = "\033[93m"
ANSI_RED = "\033[91m"

CHROMA_DB = "llama_index_collection"

PROMPT_TEMP = """
You are a helpful assistant. Use the following retrieved context to answer the user's question.
If the answer is not contained in the context, say "I don't know based on the provided information."

Context:
{context_str}

---
Answer this question based on the context above: {query_str}
"""

def extract_metadata(res_node):
    if hasattr(res_node, "metadata") and res_node.metadata:
        return res_node.metadata
    if hasattr(res_node, "node") and res_node.node is not None:
        return getattr(res_node.node, "metadata", {}) or {}
    return {}


def extract_text(res_node):
    if hasattr(res_node, "text") and res_node.text:
        return res_node.text
    if hasattr(res_node, "node") and res_node.node is not None:
        node = res_node.node
        if hasattr(node, "text") and node.text:
            return node.text
        if hasattr(node, "get_content"):
            return node.get_content()
    return ""


def build_prompt(context:str, query:str) -> PromptTemplate:
    template = PromptTemplate(template=PROMPT_TEMP)
    prompt = template.format(context_str="{context_str}", query_str="{query_str}")
    return prompt

def query_rag(database:str, persist_dir:str, data_dir:str) -> None:
    Settings.llm = Ollama(model="llama3.2:1b")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)


    node_parser = SentenceSplitter(chunk_size=250, chunk_overlap=50)
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name=database)
    except ValueError:
        pass
    chroma_collection = client.create_collection(name=database)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.node_parser = node_parser

    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Embedding {len(documents)} documents from {data_dir}... ⏳")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print(f"Database populated from {data_dir} and persisted to {persist_dir} ✅")

    retriever = VectorIndexRetriever( index=index, similarity_top_k=5)
    response_synth = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synth,
    )
    
    try:
        while True:
            user_query = input(f"\n🤖{ANSI_ORANGE}Enter your question (or type 'exit' to quit): {ANSI_RESET}")
            if user_query.lower() in ["exit", "quit"]:
                raise KeyboardInterrupt
            # Generate response
            response = query_engine.query(user_query)
            print(f"\n 🎓🦙 Response:\n {ANSI_GREEN}{response}{ANSI_RESET}")

            source_nodes = getattr(response, "source_nodes", [])
            if source_nodes:
                print("\nRetrieved context chunks:")
                for idx, res_node in enumerate(source_nodes, start=1):
                    metadata = extract_metadata(res_node)
                    file_name = (
                        metadata.get("file_name")
                        or metadata.get("filename")
                        or metadata.get("source")
                        or metadata.get("file_path")
                        or "Unknown source"
                    )
                    text = extract_text(res_node).strip()
                    print(f"{ANSI_BLUE}Chunk {idx} | {file_name}{ANSI_RESET}")
                    if text:
                        print(f"{ANSI_BLUE}{text}{ANSI_RESET}\n")
            else:
                print("\n⚠️  No source context returned for this response.")
    except KeyboardInterrupt:
        print(f"\n🦙🦙  {ANSI_RED}Exiting... Thanks for the convo!{ANSI_RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the vector database.")
    parser.add_argument(
        "--database",
        type=str,
        default=CHROMA_DB,
        help="Name of the ChromaDB collection to query.",
    )
    parser.add_argument(
        "--database_dir",
        type=str,
        default="vec_embeds",
        help="Directory where the ChromaDB collection is persisted.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing text files to be ingested.",
    )
    args = parser.parse_args()
    query_rag(database=args.database, persist_dir=args.database_dir, data_dir=args.data_dir)
