import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv() 
    st.set_page_config(page_title="AI PDF QnA Bot")
    st.header("🤖 AI PDF QnA Bot")
    
    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract text from PDF file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        
        user_prompt = st.text_input("Ask a question about your PDF:")
        response = ""
        
        if user_prompt:
            try:
                st.text("Generating answer...")
                
                docs = knowledge_base.similarity_search(user_prompt)
            
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type='stuff')
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_prompt)
                    print(cb)
                
                st.success("Answer is generated successfully")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.write(response)



if __name__ == "__main__":
    main()