from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("CREATING EMBEDDINGS")
print("=" * 60)

# Load knowledge base
df = pd.read_csv(
    "data/knowledge_base.csv"
)

print(
    f"Knowledge base size: {len(df)}"
)

# Load embedding model
print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Create embeddings
print("\nCreating embeddings...")

embeddings = model.encode(
    df["instruction"].tolist(),
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
)

print(
    f"\nEmbedding shape: "
    f"{embeddings.shape}"
)

# Create directory
os.makedirs(
    "data/embeddings",
    exist_ok=True
)

# Save embeddings
np.save(
    "data/embeddings/ticket_embeddings.npy",
    embeddings
)

# Save dataframe
df.to_pickle(
    "data/embeddings/knowledge_base.pkl"
)

print("\nSaved:")
print(
    "data/embeddings/ticket_embeddings.npy"
)

print(
    "data/embeddings/knowledge_base.pkl"
)

print("\nDONE!")