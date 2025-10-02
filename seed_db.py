
import argparse

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings

CHROMA_DB = 'vec_embeds'
DATA_PT = 'data'


def getEmbedFn():
    # embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def seed_database(data_dir: str = DATA_PT, persist_dir: str = CHROMA_DB) -> None:
    split_docs = load_and_split_documents(data_dir)
    if not split_docs:
        print(f"⚠️  No documents found under '{data_dir}'.")
        return

    chunks = assign_split_ids(split_docs)
    vector_store = create_vector_store(persist_dir)

    raw_existing = vector_store.get(include=['metadatas'])
    existing_ids = set(raw_existing.get('ids', []))
    metadata_ids = {
        meta.get('id')
        for meta in raw_existing.get('metadatas', [])
        if meta and meta.get('id')
    }
    existing_ids.update(metadata_ids)
    new_docs, new_ids = [], []
    for doc in chunks:
        doc_id = doc.metadata['id']
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if not new_docs:
        print('✅ No new documents to add.')
        return

    print(f"👉🏽 Adding {len(new_docs)} new documents to the vector store.")
    vector_store.add_documents(new_docs, ids=new_ids)



def load_and_split_documents(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(persist_directory):
    embed_fn = getEmbedFn()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function= embed_fn)
    return vector_store

def assign_split_ids(split_docs):
    last_page_id = None
    cur_split_id = 0

    for doc in split_docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        cur_page_id = f"{source}:{page}"

        if cur_page_id == last_page_id:
            cur_split_id += 1
        else:
            cur_split_id = 0
            last_page_id = cur_page_id
        doc.metadata["id"] = f"{cur_page_id}:{cur_split_id}"

    return split_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the vector database with document embeddings.")
    args = parser.parse_args()
    seed_database()
