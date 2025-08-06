import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import re
import pandas as pd
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
)
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


DOC_DIR = "documents"
INDEX_DIR = "faiss_index"
os.makedirs(DOC_DIR, exist_ok=True)


def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"} 
    )

def initialize_vectorstore(embeddings):
    if not os.path.exists(INDEX_DIR):
        print("No FAISS index found. Please process documents first.")
        return None
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def initialize_llm():
    #  for speed, e.g., "llama3:8b-q5"
    # return OllamaLLM(model="llama3:8b-q5", temperature=0.7)
    # return OllamaLLM(model="mistral", temperature=0.7)
    return OllamaLLM(model="phi3", temperature=0.7)

def load_rag_system():
    print("Initializing RAG system...")

    # Check if FAISS index exists before doing anything else
    if not os.path.exists(INDEX_DIR):
        print("No index found. You must process at least one document first.")
        return None
    
    embeddings = initialize_embeddings()
    vectorstore = initialize_vectorstore(embeddings)
    if vectorstore is None:
        return None

    llm = initialize_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversational_prompt = PromptTemplate.from_template(
        """
        You are a helpful, conversational assistant for internal support. Your job is to assist users based on the context provided (e.g., ticket logs, platform notes, internal documentation).

        If the user greets you, respond warmly **once** and ask how you can assist — do not greet repeatedly in follow-up messages.

        Always:
        - Carefully review and use the context below
        - Provide clear, friendly, and helpful responses
        - Ask follow-up questions when clarification is needed
        - Maintain an approachable, conversational tone
        - Offer guidance — never say "I don't know"

        If the user asks something unrelated to the provided context:
        - Respond helpfully, such as:
        "I'm here to help with issues related to the information I have. Do you have a question related to the context above?"
        - Avoid shutting down the conversation — keep it engaging and open

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )


    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": conversational_prompt}
    )

    return qa_chain

def process_files(file_paths):
    print("Starting document processing...")
    embeddings = initialize_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for path in file_paths:
        filename = os.path.basename(path)
        target_path = os.path.join(DOC_DIR, filename)

        if os.path.exists(target_path):
            print(f" {filename} already processed. Skipping.")
            continue

        with open(path, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        print(f"Saved: {filename}")

        if filename.endswith(".txt"):
            loader = TextLoader(target_path)
            raw_docs = loader.load()
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(target_path)
            text = df.to_string(index=False)
            raw_docs = [Document(page_content=text, metadata={"source": filename})]
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(target_path)
            raw_docs = loader.load()
        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(target_path)
            raw_docs = loader.load()
        else:
            print(f" Unsupported file type: {filename}")
            continue

        chunks = text_splitter.split_documents(raw_docs)
        print(f" {filename}: {len(chunks)} chunks created.")
        documents.extend(chunks)

    if not documents:
        print("No new documents to process.")
        return

    print(f"Updating FAISS index with {len(documents)} new documents...")
    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(INDEX_DIR)
    print("Vector DB created/updated.")

def chat_loop(qa_system):
    print("\n\033[1mYou can now chat with your documents (type 'exit' to quit):\033[0m\n")
    greeted_once = False

    def is_greeting(text):
        return re.match(r"^(hi|hello|hey|greetings|good (morning|evening))\b", text.strip(), re.I)


    while True:
        query = input("\033[1;32mUser_Query:\033[0m ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        if is_greeting(query):
            if not greeted_once:
                greeted_once = True
                print("\033[1;33mBot_Response:\033[0m Hi! How can I help you?\n")
            else:
                print("\033[1;33mBot_Response:\033[0m How can I help you?\n")
            continue

        try:
            print("\n\033[1;34mProcessing your query...\033[0m\n")
            start = time.time()
            response = qa_system.invoke({"question": query})
            end = time.time()
            answer = response.get("answer") or response.get("result") or str(response)
            print(f"\033[1;33mBot_Response:\033[0m {answer}")
            print(f"\033[90m[Took {end - start:.2f} seconds]\033[0m\n")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m\n")


if __name__ == "__main__":
    user_choice = input("Do you want to upload and process new files? (y/n): ").strip().lower()

    if user_choice == "y":
        file_input = input("Enter file paths separated by commas (e.g. C:\\file1.xlsx, C:\\file2.txt):\n> ").strip()
        files_to_process = [fp.strip() for fp in file_input.split(",") if fp.strip()]
        if not files_to_process:
            print("No valid file paths provided. Skipping document processing.")
        else:
            process_files(files_to_process)
    else:
        print("Skipping document upload. Using existing FAISS index...")

    qa_system = load_rag_system()
    if qa_system:
        chat_loop(qa_system)
