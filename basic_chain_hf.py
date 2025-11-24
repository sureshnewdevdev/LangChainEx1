# basic_chain_hf.py
"""
Simple LangChain example using a local Hugging Face model via `transformers`
and `HuggingFacePipeline`.

This does NOT call the remote Hugging Face Inference API.
No token is required.

Setup (inside your virtual environment):

    pip install "langchain>=0.3" langchain-community transformers torch

Then run:

    python basic_chain_hf.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    """Run a simple prompt -> local HF model -> string chain."""

    # 1. Choose a small causal language model
    model_name = "distilgpt2"  # small, fast, good for demos

    # 2. Load tokenizer and model from Hugging Face (downloaded once, then cached)
    print("Loading local model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. Build a text-generation pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.7,
    )

    # 4. Wrap the HF pipeline with LangChain
    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    # 5. Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful Python and AI tutor."),
            ("user", "Question: {question}"),
        ]
    )

    # 6. Output parser to convert model output into a simple string
    parser = StrOutputParser()

    # 7. Build the chain: prompt -> local model -> parse
    chain = prompt | llm | parser

    # 8. Ask a sample question
    user_question = "What is LangChain in simple words?"
    print("Q:", user_question)

    answer = chain.invoke({"question": user_question})
    print("A:", answer)


if __name__ == "__main__":
    main()
