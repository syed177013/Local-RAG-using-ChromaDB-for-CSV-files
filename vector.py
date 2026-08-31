from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import pandas as pd
import os
import time


# Configs

CSV_FILE = "dataset/myanimelist.csv"
DB_LOCATION = "./chroma_langchain_db"
COLLECTION_NAME = "anime_database"

EMBEDDING_MODEL = "mxbai-embed-large"

# Number of documents sent to Ollama per embedding request.
BATCH_SIZE = 64

# Number of results returned by the retriever.
TOP_K = 5

# Loading the dataset

print("\nLoading anime dataset...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df):,} anime records.")


# Cleaning in case we got some Nulls or empty stuff

df = df.fillna("")


# Embedding model used

print(f"\nLoading embedding model: {EMBEDDING_MODEL}")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# Initialising the VectorDB

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

# Creating Documents, for each row to hold data and related attributes chronologically in this case

documents = []
ids = []

print("\nPreparing documents...")

for _, row in df.iterrows():

    anime_id = str(row["uid"])

    title = str(row["title"]).strip()
    synopsis = str(row["synopsis"]).strip()
    genre = str(row["genre"]).strip()
    aired = str(row["aired"]).strip()

    # Text that gets embedded

    page_content = f"""
Title: {title}

Synopsis:
{synopsis}

Genre:
{genre}

Aired:
{aired}
""".strip()

    # Metadata is stored separately.

    metadata = {
        "uid": anime_id,
        "title": title,
        "genre": genre,
        "aired": aired,
        "episodes": str(row["episodes"]),
        "members": str(row["members"]),
        "popularity": str(row["popularity"]),
        "ranked": str(row["ranked"]),
        "score": str(row["score"]),
        "link": str(row["link"]),
    }

    document = Document(
        page_content=page_content,
        metadata=metadata
    )

    documents.append(document)
    ids.append(anime_id)


print(f"Prepared {len(documents):,} documents.")


# Check for existing DB so we dont have to re vectorise again

print("\nChecking existing vector database...")

existing_ids = set()

try:
    existing_data = vector_store.get(include=[])

    if existing_data and existing_data.get("ids"):
        existing_ids = set(existing_data["ids"])

except Exception as e:
    print(f"Could not inspect existing database: {e}")


# Only embed documents that are not already present.
new_documents = []
new_ids = []

for document, document_id in zip(documents, ids):

    if document_id not in existing_ids:
        new_documents.append(document)
        new_ids.append(document_id)


print(f"Already indexed: {len(existing_ids):,}")
print(f"New documents:   {len(new_documents):,}")


# Embed + Insert

if new_documents:

    print("\nStarting vectorisation...")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)

    total = len(new_documents)
    start_time = time.time()

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_documents = new_documents[start:end]
        batch_ids = new_ids[start:end]

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )

        elapsed = time.time() - start_time

        processed = end
        percentage = (processed / total) * 100

        rate = processed / elapsed if elapsed > 0 else 0

        remaining = total - processed

        eta = remaining / rate if rate > 0 else 0

        print(
            f"\rProgress: {processed:,}/{total:,} "
            f"({percentage:6.2f}%) | "
            f"{rate:.1f} docs/s | "
            f"ETA: {eta / 60:.1f} min",
            end="",
            flush=True
        )

    elapsed = time.time() - start_time

    print("\n")
    print("-" * 60)
    print(
        f"Finished embedding {total:,} documents "
        f"in {elapsed / 60:.2f} minutes."
    )

else:

    print("\nVector database is already up to date.")


# retrieve  

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)

print("\nVector database ready.")