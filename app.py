import langchain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
# import apiKeys
import os

import apiKeys

print("Files available for processing:")
dir = 'FILE'
filedir = {}
print("INDEX \t FILENAME")
for idx, files in enumerate(os.listdir(dir)):
    if files == "PLACE_YOUR_FIE_HERE.txt":
        continue
    print(f"{idx + 1} \t {files}")
    filedir.update({idx + 1: 'FILE/' + f'{files}'})
choice = int(input("Enter the index of the file to be processed: "))
PATH = filedir[choice]
print(PATH)

data = UnstructuredPDFLoader(PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=3)
texts = chunks.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key=apiKeys.OPENAI_API_KEY)

pinecone.init(
    api_key=apiKeys.PINECONE_API_KEY,
    environment=apiKeys.PINECONE_API_ENVIRONMENT
)

index = apiKeys.INDEX_NAME

buffer = Pinecone.from_texts([text.page_content for text in texts], embeddings, index_name=index)

init_llm = OpenAI(temperature=0, openai_api_key=apiKeys.OPENAI_API_KEY)
chain = load_qa_chain(init_llm, chain_type="stuff")

query = input("Enter your query")

docs_to_search_from = buffer.similarity_search(query)

answer = chain.run(input_documents=docs_to_search_from, question=query)

print(answer)
