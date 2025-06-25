import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()
#os.environ("hf_token") = os.getenv("HuggingFace_API_KEY")
os.environ["hf_token"] = os.getenv("HuggingFace_API_KEY")
os.environ["LangChain_API_KEY"] = os.getenv("LangChain_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name = "all-miniLM-L6-v2")
# groq_api_key = os.getenv("GROQ_API_KEY")
 # os.getenv("GROQ_API_KEY")

# set up streamlit
st.title("conversational RAG with PDF Upload and Chat history")
st.write("Upload the pdf's and chat with their content ")

# create a GroqAPIKEY
api_key = st.text_input("enter you GROQ API KEY:", type = "password")

## checking wether the api key is given are not
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")
    # chat interface
    session_id = st.text_input("Session ID", value="deault_session")
    
    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("choose a PDF file", type = "pdf", accept_multiple_files=True)
    
    ## process upload file PDF's 
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.read())
                file_name = uploaded_file.name
            
            # load the pdf
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
        # split and create embeddings for the documents
        text_spliter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
        split_doc = text_spliter.split_documents(documents)
        vectorstore = Chroma.from_documents(embedding=embeddings,documents=split_doc)
        retrievers = vectorstore.as_retriever()
    
        # teaching the q uestion prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "Which might reference context in the chat history"
            "formulate a standalone questions which can be understood"
            "just reformulate it if needed and otherwise return it as it."
        )
            
        # crete a prompt 
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human","{input}")
            ]
        )
        
        # calling the history retriever (model, storedb_embedding_retrievers, prompt_history)
        history_aware_retriever = create_history_aware_retriever(llm, retrievers, contextualize_q_prompt)
        
        ## answer question
        system_promt = (
            "You are an assistant for question-answering tasks."
            "use the following pieces of retrieved context to answer"
            "the question. If you dont know the answer, say that you"
            "don't know. Use three sentences maximun and keep the"
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        QA_prompt = ChatPromptTemplate(
            [
                ("system",system_promt),
                MessagesPlaceholder(variable_name = "chat_history"),
                ("human","{input}")
            ]
        )
        
        # create the question answer chain
        question_answer_chain = create_stuff_documents_chain(llm,QA_prompt)
        RAG_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            # get the session history
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        
        conversational_rag_chain = RunnableWithMessageHistory(
            RAG_chain,
            get_session_history,  # must return a `BaseChatMessageHistory` instance
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input = st.text_input("enter your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id": session_id}
                }
            )
            
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer']) 
            st.write("Chat history:",session_history.messages)

else:
    st.warning("please the key API_key or please enter the correct API_KEY")