import os
import shutil
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from model.format import load_json_documents

def load_compliance_rules(file_path="compliance_check.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Compliance rules file not found at {file_path}")
        return ""

def reset_and_create_vectorstore(documents, embedding, persist_dir="chroma_store"):
    """Deletes any existing Chroma DB and creates a new one fresh."""
    # Delete old vectorstore folder
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("🧹 Old Chroma store cleared.")

    # Create fresh Chroma vectorstore from documents
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_dir
    )
    print("✅ Fresh Chroma vectorstore created for this session.")

    return vectordb

def main():
    print("⚙️ Starting fresh LLM-based QA session...\n")

    folder_path = "output_folder"
    persist_dir = "chroma_store"

    # Load documents
    documents = load_json_documents(folder_path)
    print(f"✅ Loaded {len(documents)} documents from JSON files.")

    if not documents:
        print("❌ No documents found. Check your folder path.")
        return

    # Load compliance rules
    compliance_rules = load_compliance_rules()
    if not compliance_rules:
        print("❌ Compliance rules missing or empty. Please add the file.")
        return

    # Initialize LLM and Embeddings
    llm = OllamaLLM(model="llama3.1")
    embedding = OllamaEmbeddings(model="llama3.1")

    # Reset vectorstore and create from current documents
    vectordb = reset_and_create_vectorstore(documents, embedding, persist_dir=persist_dir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 9})

    # Set up QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    system_prefix = """
You are a compliance assistant. Extract factual details from the given documents only.
Do not guess or hallucinate. If you find contradicting values (e.g. different customer names), clearly state it.
Always refer to the specific document source where the value was found.
"""

    print("🔍 Ready! Ask any compliance or document query. Type 'exit' to quit.\n")

    while True:
        user_query = input("🧠 Query > ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("👋 Exiting and resetting vector database.")
            # Optional: clean up vectorstore on exit
            shutil.rmtree(persist_dir, ignore_errors=True)
            break
        if not user_query:
            continue

        try:
            if "compliance" in user_query.lower():
                prompt = system_prefix + "\n\n" + compliance_rules + "\n\nUser Query: " + user_query
            else:
                prompt = system_prefix + "\n\nUser Query: " + user_query

            docs = retriever.invoke(prompt)
            print(f"📦 Retrieved {len(docs)} relevant documents:")
            for doc in docs:
                print(f" - Source: {doc.metadata.get('source')}")

            answer = qa.invoke(prompt)
            print("\n📄 Result:\n" + answer["result"] + "\n")

        except Exception as e:
            print(f"❌ Error answering query: {e}")

if __name__ == "__main__":
    main()
