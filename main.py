
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
import os

import streamlit as st

file=open("D:\INeuron\pdfqa/Napolean Bonaparte History.txt","r")

doc=file.read()

text=""

for txt in doc:

    text+=txt



text_splitter=CharacterTextSplitter(
    separator="""\n""",
    chunk_size=200,
    chunk_overlap=50,
    length_function=len
)

data=text_splitter.split_text(text)

os.environ["cohere_api_key"]="------------"

embedding=CohereEmbeddings()
llm=Cohere(temperature=0.8)

vectorStore=FAISS.from_texts(data,embedding)

st.header("Learn Napolean Bonaparte History")
st.subheader("Musician9DX")

text_input=st.text_input("Enter Query")

ragChain=load_qa_chain(llm)

if text_input:

    paragraph=vectorStore.similarity_search(text_input)
    text=ragChain.run(input_documents=paragraph,question=text_input)
    text=st.write(text)








