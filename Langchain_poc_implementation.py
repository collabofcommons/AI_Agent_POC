# Langchain Embedding Example
# Author: Gareth Sharpe
# Description: An example using the Langchain client, embeddings, and the Confluence document loader.

# Install our Dependencies
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import ConfluenceLoader

load_dotenv()

# Initialize my LLM
print("Initializing LLM...")
llm = AzureChatOpenAI(deployment_name="gpt-4-0")

# Load Confluence space
print("Loading articles...")
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key="CCTODIOOK",
    include_attachments=False,
    include_archived_content=False,
    include_comments=False,
    include_restricted_content=False,
    cloud=True,
)
documents = loader.load()

# Initialize Azure embedding
print("Initializing Azure embedding...")
embedding = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")

# Index my content
print("Indexing.. This could take a while. Especially if you have a Windows machine. LOL.")
index_creator = VectorstoreIndexCreator(embedding=embedding)
index = index_creator.from_documents(documents)

# Provide initial query
query = "How do I make a cup of coffee?"
response = index.query_with_sources(query, llm=llm)

print(response)

# Print the response
print("Question:", response["question"])
print("Answer:", response["answer"])
print("Sources:", response["sources"])

# Let the user ask a question
question = input("Question: ")
response = index.query_with_sources(question, llm=llm)

# Print the response
print("Answer:", response["answer"])
print("Sources:", response["sources"])
