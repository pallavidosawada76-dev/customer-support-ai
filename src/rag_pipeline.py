from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from router import route_ticket


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

data = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

X = [row["instruction"] for row in data]
y_category = [row["category"] for row in data]
y_intent = [row["intent"] for row in data]

print("Total examples:", len(X))


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

(
    X_train,
    X_test,
    y_category_train,
    y_category_test,
    y_intent_train,
    y_intent_test
) = train_test_split(
    X,
    y_category,
    y_intent,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)


# ============================================================
# 3. CATEGORY TF-IDF
# ============================================================

print("\nCreating TF-IDF...")

category_vectorizer = TfidfVectorizer()

X_train_tfidf = category_vectorizer.fit_transform(
    X_train
)


# ============================================================
# 4. TRAIN CATEGORY MODEL
# ============================================================

print("Training category model...")

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train_tfidf,
    y_category_train
)


# ============================================================
# 5. BUILD CATEGORY → INTENT MAPPING
# ============================================================

category_intents = {}

for category, intent in zip(
    y_category_train,
    y_intent_train
):

    if category not in category_intents:
        category_intents[category] = set()

    category_intents[category].add(intent)


# ============================================================
# 6. TRAIN INTENT MODEL FOR EACH CATEGORY
# ============================================================

print("\nTraining category-specific intent models...")

intent_models = {}

for category in sorted(category_intents):

    category_X = []
    category_y = []

    for text, cat, intent in zip(
        X_train,
        y_category_train,
        y_intent_train
    ):

        if cat == category:

            category_X.append(text)
            category_y.append(intent)


    unique_intents = set(category_y)


    # --------------------------------------------------------
    # Only one intent
    # --------------------------------------------------------

    if len(unique_intents) == 1:

        only_intent = list(unique_intents)[0]

        intent_models[category] = {
            "type": "single",
            "intent": only_intent
        }

        continue


    # --------------------------------------------------------
    # Multiple intents
    # --------------------------------------------------------

    intent_vectorizer = TfidfVectorizer()

    category_X_tfidf = intent_vectorizer.fit_transform(
        category_X
    )

    intent_model = LogisticRegression(
        max_iter=1000
    )

    intent_model.fit(
        category_X_tfidf,
        category_y
    )

    intent_models[category] = {
        "type": "model",
        "model": intent_model,
        "vectorizer": intent_vectorizer
    }


# ============================================================
# 7. LOAD KNOWLEDGE BASE
# ============================================================

print("\nLoading knowledge base...")

knowledge_base = pd.read_pickle(
    "data/embeddings/knowledge_base.pkl"
)

print(
    "Knowledge base records:",
    len(knowledge_base)
)


# ============================================================
# 8. LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    "data/embeddings/ticket_embeddings.npy"
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# 9. LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# 10. PROCESS CUSTOMER TICKET
# ============================================================

def process_ticket(ticket):

    # --------------------------------------------------------
    # STEP 1 — CATEGORY
    # --------------------------------------------------------

    ticket_tfidf = category_vectorizer.transform(
        [ticket]
    )

    predicted_category = category_model.predict(
        ticket_tfidf
    )[0]


    # --------------------------------------------------------
    # STEP 2 — INTENT
    # --------------------------------------------------------

    selected_model = intent_models[
        predicted_category
    ]


    if selected_model["type"] == "single":

        predicted_intent = selected_model[
            "intent"
        ]

    else:

        intent_vectorizer = selected_model[
            "vectorizer"
        ]

        intent_model = selected_model[
            "model"
        ]

        ticket_intent_tfidf = (
            intent_vectorizer.transform(
                [ticket]
            )
        )

        predicted_intent = intent_model.predict(
            ticket_intent_tfidf
        )[0]


    # --------------------------------------------------------
    # STEP 3 — ROUTING
    # --------------------------------------------------------

    assigned_team = route_ticket(
        predicted_category
    )


    # --------------------------------------------------------
    # STEP 4 — CREATE EMBEDDING
    # --------------------------------------------------------

    ticket_embedding = embedding_model.encode(
        [ticket]
    )


    # --------------------------------------------------------
    # STEP 5 — FILTER KNOWLEDGE BASE
    # --------------------------------------------------------

    valid_indices = knowledge_base.index[
        (knowledge_base["category"] == predicted_category)
        &
        (knowledge_base["intent"] == predicted_intent)
    ].to_numpy()


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if len(valid_indices) == 0:

        valid_indices = knowledge_base.index[
            knowledge_base["category"]
            == predicted_category
        ].to_numpy()


    # --------------------------------------------------------
    # STEP 6 — CALCULATE SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        ticket_embedding,
        embeddings[valid_indices]
    )[0]


    # --------------------------------------------------------
    # STEP 7 — TOP 3 RESULTS
    # --------------------------------------------------------

    top_k = min(
        3,
        len(valid_indices)
    )

    top_positions = np.argsort(
        similarities
    )[-top_k:][::-1]


    # --------------------------------------------------------
    # STEP 8 — BUILD RETRIEVED RESULTS
    # --------------------------------------------------------

    retrieved_tickets = []

    for position in top_positions:

        index = valid_indices[position]

        retrieved_tickets.append({

            "instruction":
                knowledge_base.iloc[index]["instruction"],

            "category":
                knowledge_base.iloc[index]["category"],

            "intent":
                knowledge_base.iloc[index]["intent"],

            "response":
                knowledge_base.iloc[index]["response"],

            "similarity":
                similarities[position]

        })


    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        "ticket": ticket,

        "category": predicted_category,

        "intent": predicted_intent,

        "assigned_team": assigned_team,

        "retrieved_tickets": retrieved_tickets

    }


# ============================================================
# 11. TEST
# ============================================================

test_ticket = (
    "I was charged twice for my order"
)

result = process_ticket(
    test_ticket
)


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RAG PIPELINE")
print("=" * 60)

print("\nCustomer Ticket:")
print(result["ticket"])

print("\nPredicted Category:")
print(result["category"])

print("\nPredicted Intent:")
print(result["intent"])

print("\nAssigned Team:")
print(result["assigned_team"])


print("\n")
print("=" * 60)
print("RETRIEVED SIMILAR TICKETS")
print("=" * 60)


for i, ticket in enumerate(
    result["retrieved_tickets"],
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        "Similarity:",
        round(
            ticket["similarity"],
            4
        )
    )

    print(
        "Ticket:",
        ticket["instruction"]
    )

    print(
        "Category:",
        ticket["category"]
    )

    print(
        "Intent:",
        ticket["intent"]
    )

    print(
        "Previous Response:",
        ticket["response"]
    )