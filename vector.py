from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document # This is used to create documents from the reviews
import  os
import pandas as pd
from nltk.corpus.reader import documents

df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# location of the database that will be created
db_location= "./chroma_langchain_db"
# check if the database already exists
db_exist = os.path.exists(db_location)

if  not db_exist:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(page_content=row["Title"] + " " + row["Review"],
                                  metadata={"rating": row["Rating"], "date": row["Date"]},# We don't need this # metadata for now, but we can use it later
                                  id = str(i))
        documents.append(document)
        ids.append(str(i))

# create the vector store and persist it to the database
vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
    persist_directory=db_location, # store it permanently in the database
    )

if not db_exist:
    vector_store.add_documents(documents = documents, ids=ids)

# connecting the vector store to the LLM (looking for the most relevant documents and passing them to the LLM)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # retrieve the top 5 most relevant documents
