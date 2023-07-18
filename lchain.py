import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PyPDF2 import PdfReader
import pickle
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY") 

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory= ConversationBufferWindowMemory(k=3)

def main():
    
    
    st.header("Chat With Your Documents💬")

    with st.sidebar:
        st.image("./image/summarify-logo-300.png")
        "contacts"
        st.text("http:/summarify.io")
        
    pdf = st.file_uploader("Upload your PDF file here", accept_multiple_files=False)
    
    temperature = st.slider(
    'How much our AI model should be deterministic?',
    0, 10, 1)   
    st.caption("0 recommended")
    
    llm = OpenAI(temperature=temperature/10,model_name="gpt-3.5-turbo",openai_api_key=key)       
    
    
    if pdf is not None:
        name = pdf.name

        if not os.path.exists(name):

            readPDF = PdfReader(pdf)
            text = ""

            question = st.text_input("Enter",key="input...")
            for page in readPDF.pages:
                text += page.extract_text()
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2800,
                chunk_overlap=50,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            vectorStore = FAISS.from_documents(chunks,embedding=embeddings)

            with open(f'{name}.pkl', 'wb') as file:
                pickle.dump(textSearch, file)

            qa = RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=vectorStore.as_retriever(), memory=st.session_state.buffer_memory)

            st.write("---")

            st.write(qa.run(question))

            st.write("---")

        else:

                readPDF = PdfReader(pdf)
                text = ""

                question = st.text_input("Enter",key="input...")
                for page in readPDF.pages:
                    text += page.extract_text()
                
                with open("./langchain.cpython-39.pyc","rb") as file:
                    textSearch = pickle.load(file)

                qa = RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=textSearch.as_retriever(), memory=st.session_state.buffer_memory)

                st.write("---")

                st.write(qa.run(question))

                st.write("---")
            

        
        
            
        
        
        
        
        
    
if __name__ == "__main__":
    main()






