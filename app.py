import streamlit as st
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG With PDF")
#add groq api key through streamlit side bar
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if 'store' not in st.session_state:
    st.session_state.store = {}

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    
    if uploaded_files:
       
        if "retriever" not in st.session_state:
            with st.spinner("Processing your PDFs..."):
                documents = []
                for uploaded_file in uploaded_files:
                    
                    temp_path = f"./{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                    os.remove(temp_path) # Clean up

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                
                st.session_state.retriever = vectorstore.as_retriever()
                st.success("Retriever Ready!")

    
    if "retriever" in st.session_state:
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        
        def get_contextualized_docs(input_dict):
           
            if "retriever" not in st.session_state:
                return []
            
            if input_dict.get("chat_history"):
                standalone_q = history_aware_retriever_chain.invoke(input_dict)
                return st.session_state.retriever.invoke(standalone_q)
            
            return st.session_state.retriever.invoke(input_dict["input"])

        
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(get_contextualized_docs(x))
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        
        def get_session_history(sid: str):
            if sid not in st.session_state.store:
                st.session_state.store[sid] = ChatMessageHistory()
            return st.session_state.store[sid]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        
        user_input = st.chat_input("Type your message here...")
        if user_input:
            
            full_response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(f"**Assistant:** {full_response}")

            
            with st.expander("Show Chat History"):
                for msg in st.session_state.store[session_id].messages:
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    st.write(f"**{role}:** {msg.content}")

    else:
        st.info("Waiting for PDF upload and processing...")
else:
    st.warning("Please provide your Groq API Key.")
