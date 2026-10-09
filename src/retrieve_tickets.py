from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD KNOWLEDGE BASE
# ============================================================

print("Loading knowledge base...")

df = pd.read_pickle(
    "data/embeddings/knowledge_base.pkl"
)

print("Knowledge base records:", len(df))


# ============================================================
# 2. LOAD SAVED EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    "data/embeddings/ticket_embeddings.npy"
)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 3. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded!")


# ============================================================
# 4. RETRIEVAL FUNCTION
# ============================================================

def retrieve_similar_tickets(
    query,
    category=None,
    top_k=3
):

    # --------------------------------------------------------
    # Convert the new ticket into an embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query]
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]


    # --------------------------------------------------------
    # Filter knowledge base by category
    # --------------------------------------------------------

    if category is not None:

        valid_indices = df.index[
            df["category"] == category
        ].to_numpy()

    else:

        valid_indices = np.arange(
            len(df)
        )


    # --------------------------------------------------------
    # Get similarity scores for valid tickets
    # --------------------------------------------------------

    category_similarities = similarities[
        valid_indices
    ]


    # --------------------------------------------------------
    # Find top K most similar tickets
    # --------------------------------------------------------

    top_positions = np.argsort(
        category_similarities
    )[-top_k:][::-1]


    top_indices = valid_indices[
        top_positions
    ]


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = []

    for index in top_indices:

        results.append({

            "instruction":
                df.iloc[index]["instruction"],

            "category":
                df.iloc[index]["category"],

            "intent":
                df.iloc[index]["intent"],

            "response":
                df.iloc[index]["response"],

            "similarity":
                similarities[index]

        })


    return results


# ============================================================
# 5. TEST QUERY
# ============================================================

query = "I was charged twice for my order"


# ============================================================
# 6. CATEGORY FILTER
# ============================================================

# For this test we already know the ticket is PAYMENT.
#
# Later we will get this automatically from our
# category classification model.

predicted_category = "PAYMENT"


# ============================================================
# 7. RETRIEVE SIMILAR TICKETS
# ============================================================

results = retrieve_similar_tickets(
    query,
    category=predicted_category,
    top_k=3
)


# ============================================================
# 8. DISPLAY NEW TICKET
# ============================================================

print("\n==========================================")
print("NEW CUSTOMER TICKET")
print("==========================================")

print(query)

print("\nPredicted Category:", predicted_category)


# ============================================================
# 9. DISPLAY RETRIEVED TICKETS
# ============================================================

print("\n==========================================")
print("TOP 3 SIMILAR TICKETS")
print("==========================================")


for i, result in enumerate(
    results,
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        "Similarity:",
        round(
            result["similarity"],
            4
        )
    )

    print(
        "Ticket:",
        result["instruction"]
    )

    print(
        "Category:",
        result["category"]
    )

    print(
        "Intent:",
        result["intent"]
    )

    print(
        "Previous Response:",
        result["response"]
    )