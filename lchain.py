import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader


os.environ["OPENAI_API_KEY"] = "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"

template =  """ Given the text: "{query}".

    -Based on the provided text, generate 5 questions and answer options (a,b,c and d):
    -Generate these questions with respect to context of the given text input

"""

         
prompt = PromptTemplate(
    template=template,
    input_variables=["query"] 
)                                                         

def main():
    
    
    st.header("Generate Questions From Your PDF")
    with st.sidebar:
        st.image("./image/summarify-logo-300.png")
        
        "contacts"
        st.text("http:/summarify.io")
        
    pdf = st.file_uploader("Upload your PDF file here", accept_multiple_files=False)
    
    temperature = st.slider(
    'How much our AI model should be deterministic?',
    0, 10, 1)   
    st.caption("0 recommended")
    
    llm = OpenAI(temperature=temperature/10,model_name="text-davinci-003",openai_api_key=os.getenv("OPENAI_API_KEY"))       
    
    
    if pdf is not None:
        
        readPDF = PdfReader(pdf)
        text = ""
        
        for page in readPDF.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2800,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write("---")
        st.write(llm(prompt.format(query=chunks[(int)(len(chunks)/2):5])))
        st.write("---")
        
        
            
        
        
        
        
        
    
if __name__ == "__main__":
    main()






