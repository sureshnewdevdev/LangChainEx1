from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("Downloading embedding model...")
model = SentenceTransformer(model_name)

print("Done! Model is cached locally in the Hugging Face cache folder.")
