import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

def load_and_process_webpage(url):
    """
    Loads webpage text, splits it into chunks, generates embeddings,
    and stores them in a persistent Chroma vector store.
    """
    print(f"Loading: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print("Generating embeddings for each chunk using Ollama LLaMA3...")
    embeddings = OllamaEmbeddings(model="llama3")

    print("Storing embeddings in Chroma vector store...")
    # Persist directory updated per URL could be added, but here we overwrite for simplicity
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()   

    return vectorstore

def build_qa_chain(vectorstore):
    """
    Creates a RetrievalQA chain with:
    - retriever from the vectorstore that fetches top-k relevant chunks
    - LLM from Ollama LLaMA3 to generate answers based on retrieved chunks
    """
    print("Building QA chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    # Initial URL input (no mention of JS rendering)
    url = input("Enter a webpage URL: ").strip()
    vectorstore = load_and_process_webpage(url)
    qa_chain = build_qa_chain(vectorstore)

    print("\nAsk questions about the webpage content.")
    print("Type 'NEW' (all caps) to load another webpage.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        elif question == "NEW":
            # Load a new webpage and rebuild vectorstore + QA chain
            url = input("Enter a new webpage URL: ").strip()
            vectorstore = load_and_process_webpage(url)
            qa_chain = build_qa_chain(vectorstore)
            continue

        # Run question through the QA chain to get answer
        answer = qa_chain.run(question)

        # Simple relevance check:
        # If answer looks like it didn't find relevant info, respond accordingly.
        # This is a basic heuristic; for production, use similarity score or other metric.
        if answer.strip() == "" or "I don't know" in answer.lower() or "cannot find" in answer.lower():
            print("This topic is irrelevant. Please ask another question.")
        else:
            print("Answer:", answer)

if __name__ == "__main__":
    main()