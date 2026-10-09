import pandas as pd
import os

print("=" * 60)
print("CREATING KNOWLEDGE BASE")
print("=" * 60)

# Load augmented dataset
df = pd.read_csv(
    "data/augmented_dataset.csv"
)

# Keep only the fields needed for RAG
knowledge_base = df[
    [
        "instruction",
        "category",
        "intent",
        "response"
    ]
].copy()

# Save knowledge base
knowledge_base.to_csv(
    "data/knowledge_base.csv",
    index=False
)

print(
    f"Knowledge base created: "
    f"{len(knowledge_base)} records"
)

print("\nColumns:")
print(
    knowledge_base.columns.tolist()
)

print("\nSample:")
print(
    knowledge_base.head()
)

print("\nSaved to:")
print(
    "data/knowledge_base.csv"
)