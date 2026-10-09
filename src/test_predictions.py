from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
# 3. TRAIN CATEGORY MODEL
# ============================================================

print("\nTraining category model...")

category_vectorizer = TfidfVectorizer()

X_train_tfidf = category_vectorizer.fit_transform(
    X_train
)

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train_tfidf,
    y_category_train
)


# ============================================================
# 4. BUILD CATEGORY → INTENT MAPPING
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
# 5. TRAIN INTENT MODEL FOR EACH CATEGORY
# ============================================================

print("\nTraining intent models...")

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


    # Category has only one intent
    if len(unique_intents) == 1:

        only_intent = list(unique_intents)[0]

        intent_models[category] = {
            "type": "single",
            "intent": only_intent
        }

        continue


    # Category has multiple intents
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
# 6. PREDICTION FUNCTION
# ============================================================

def predict_ticket(ticket):

    # -----------------------------
    # Predict category
    # -----------------------------

    ticket_tfidf = category_vectorizer.transform(
        [ticket]
    )

    category = category_model.predict(
        ticket_tfidf
    )[0]


    # -----------------------------
    # Predict intent inside category
    # -----------------------------

    selected_model = intent_models[category]

    if selected_model["type"] == "single":

        intent = selected_model["intent"]

    else:

        intent_vectorizer = selected_model["vectorizer"]
        intent_model = selected_model["model"]

        ticket_intent_tfidf = (
            intent_vectorizer.transform([ticket])
        )

        intent = intent_model.predict(
            ticket_intent_tfidf
        )[0]


    # -----------------------------
    # Route
    # -----------------------------

    team = route_ticket(category)


    return category, intent, team


# ============================================================
# 7. TEST TICKETS
# ============================================================

test_tickets = [

    # PAYMENT
    "My card was charged twice",

    "I was charged twice for the same transaction",

    "My payment failed",

    "I cannot complete my payment",

    "What payment methods do you accept?",


    # ORDER
    "Where is my order?",

    "I want to track my order",

    "I want to cancel my order",

    "I want to change my order",

    "I want to place a new order",


    # ACCOUNT
    "I forgot my password",

    "I cannot log into my account",

    "I want to delete my account",

    "I want to create a new account",


    # REFUND
    "I want a refund",

    "How can I get my money back?",

    "What is your refund policy?",


    # SHIPPING
    "I want to change my shipping address",

    "Where can I set my shipping address?",


    # DELIVERY
    "Where is my delivery?",

    "How long will delivery take?"

]


# ============================================================
# 8. RUN PREDICTIONS
# ============================================================

print("\n")
print("=" * 75)
print("CLASSIFIER STRESS TEST")
print("=" * 75)


for ticket in test_tickets:

    category, intent, team = predict_ticket(
        ticket
    )

    print("\nTicket:")
    print(ticket)

    print("Category:", category)
    print("Intent:", intent)
    print("Team:", team)

    print("-" * 75)