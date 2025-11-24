from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Done! Model is cached locally in the Hugging Face cache folder.")
