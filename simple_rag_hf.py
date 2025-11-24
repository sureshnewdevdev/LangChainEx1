# simple_rag_hf.py
"""
Simple RAG (Retrieve + Generate) example using local Hugging Face models:

- Local text-generation model via `transformers` + `HuggingFacePipeline`
- Local sentence-transformers embeddings via `HuggingFaceEmbeddings`
- FAISS as in-memory vector store
- A local text file (data/notes.txt) as the knowledge base

NO Hugging Face Inference API and NO token required.

-----------------------------------------
Setup (inside your virtual environment):
-----------------------------------------

1. Install dependencies:

    pip install "langchain>=0.3" langchain-community langchain-text-splitters \
        transformers sentence-transformers faiss-cpu torch

2. Make sure you have a file:

    data/notes.txt

3. Run:

    python simple_rag_hf.py

Then type your questions at the prompt.
"""

import os

# ------------------ Transformers / HF models ------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------ LangChain core & community ----------------
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -------------------------------------------------------------------
# Helper: format retrieved documents into a single context string
# -------------------------------------------------------------------
def format_docs(docs):
    """
    Convert a list of Document objects into one long context string.

    Each Document has a .page_content attribute that stores the text.
    We join them with blank lines so the LLM sees them as separate chunks.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------------------------------------------------
# Build the RAG chain (Retrieve + Generate)
# -------------------------------------------------------------------
def build_rag_chain():
    """
    Build a simple RAG pipeline using only local models.

    Steps:
    1. Read the knowledge base from data/notes.txt
    2. Split it into smaller text chunks
    3. Turn chunks into embeddings and store them in a FAISS index
    4. Create a retriever over the FAISS index
    5. Define a prompt template that takes {context} + {question}
    6. Use a local HF text-generation model as the LLM
    7. Combine everything into a LangChain runnable (rag_chain)
    """

    # 1. Read our local knowledge base
    data_path = os.path.join("data", "notes.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Knowledge base file not found at: {data_path}\n"
            "Create data/notes.txt with your notes before running this script."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    if not full_text.strip():
        raise ValueError("data/notes.txt is empty. Add some content and try again.")

    # 2. Split the text into manageable overlapping chunks
    #    - chunk_size: max characters per chunk
    #    - chunk_overlap: overlap between chunks to preserve context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError(
            "No text chunks were created from notes.txt. "
            "Check that the file actually contains text."
        )

    # 3. Local embeddings using sentence-transformers
    #    This downloads the model the first time and then caches it.
    #    all-MiniLM-L6-v2 is small and fast for demos.
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 4. Create a FAISS vector store from the text chunks
    #    FAISS stores embeddings and allows efficient similarity search.
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # 5. Turn the vector store into a retriever
    #    search_kwargs={"k": 3} means: return top 3 similar chunks for each query.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 6. Define the RAG prompt template
    #    The LLM will see the context and the question together.
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions only using the provided context.
If the answer is not in the context, say "I don't know from the given notes."

Context:
{context}

Question:
{question}
"""
    )

    # 7. Define the local LLM via transformers
    #    distilgpt2 is small and works well enough for demo purposes.
    llm_model_name = "distilgpt2"

    print(f"Loading local LLM: {llm_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)

    # Build a text-generation pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,   # how many tokens to generate
        temperature=0.7,      # creativity / randomness
    )

    # Wrap the pipeline so LangChain can treat it as an LLM
    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    # 8. Output parser: converts the model output to a string
    parser = StrOutputParser()

    # 9. Build the RAG chain using LangChain's runnable composition
    #
    # RunnableParallel:
    #   - Takes the user question as input
    #   - context = retriever(question) -> format_docs -> string
    #   - question = the same raw question (RunnablePassthrough)
    #
    # Then we pipe the result into:
    #   - prompt (fills {context} and {question})
    #   - llm (local model)
    #   - parser (clean string output)
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | parser
    )

    return rag_chain


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------
def main():
    # Build the RAG pipeline once at startup
    rag_chain = build_rag_chain()

    print("Simple LangChain RAG demo (Local Hugging Face Models)")
    print("Your knowledge base is: data/notes.txt")
    print("Ask a question about its content.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_q = input("Your question: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_q:
            print("Please type a non-empty question.\n")
            continue

        # Invoke the RAG chain with the user question
        try:
            answer = rag_chain.invoke(user_q)
        except Exception as e:
            print("\n[Error while generating answer]", e)
            print("-" * 60)
            continue

        print("\nAnswer:\n", answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
