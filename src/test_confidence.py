from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = pd.DataFrame(dataset)

X = df["instruction"]
y_category = df["category"]
y_intent = df["intent"]


# ==================================================
# 2. SPLIT DATA
# ==================================================

(
    X_train,
    X_test,
    y_cat_train,
    y_cat_test,
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


# ==================================================
# 3. CATEGORY MODEL
# ==================================================

print("\nTraining category model...")

category_vectorizer = TfidfVectorizer()

X_train_cat = category_vectorizer.fit_transform(X_train)

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train_cat,
    y_cat_train
)


# ==================================================
# 4. CATEGORY → INTENT MAPPING
# ==================================================

category_intents = {}

for category, intent in zip(
    y_cat_train,
    y_intent_train
):

    if category not in category_intents:
        category_intents[category] = set()

    category_intents[category].add(intent)


# ==================================================
# 5. TRAIN INTENT MODELS
# ==================================================

intent_models = {}
intent_vectorizers = {}

print("\nTraining intent models...")

for category, intents in category_intents.items():

    if len(intents) == 1:
        continue

    mask = y_cat_train == category

    category_text = X_train[mask]
    category_intent = y_intent_train[mask]

    vectorizer = TfidfVectorizer()

    X_category = vectorizer.fit_transform(
        category_text
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_category,
        category_intent
    )

    intent_vectorizers[category] = vectorizer
    intent_models[category] = model


# ==================================================
# 6. PREDICTION FUNCTION
# ==================================================

def predict_ticket(ticket):

    # ------------------------------
    # Category
    # ------------------------------

    category_vector = category_vectorizer.transform(
        [ticket]
    )

    category_probabilities = (
        category_model.predict_proba(
            category_vector
        )[0]
    )

    category_index = category_probabilities.argmax()

    category = category_model.classes_[
        category_index
    ]

    category_confidence = (
        category_probabilities[
            category_index
        ]
    )


    # ------------------------------
    # Intent
    # ------------------------------

    if category not in intent_models:

        intent = list(
            category_intents[category]
        )[0]

        intent_confidence = 1.0

    else:

        vectorizer = intent_vectorizers[
            category
        ]

        model = intent_models[
            category
        ]

        ticket_vector = vectorizer.transform(
            [ticket]
        )

        intent_probabilities = (
            model.predict_proba(
                ticket_vector
            )[0]
        )

        intent_index = (
            intent_probabilities.argmax()
        )

        intent = model.classes_[
            intent_index
        ]

        intent_confidence = (
            intent_probabilities[
                intent_index
            ]
        )


    return (
        category,
        category_confidence,
        intent,
        intent_confidence
    )


# ==================================================
# 7. TEST TICKETS
# ==================================================

tickets = [

    "I forgot my password",

    "I want to cancel my order",

    "Where is my order?",

    "What payment methods do you accept?",

    "My payment failed",

    "My card was charged twice",

    "There is a duplicate charge on my card",

    "There is an extra charge for my purchase",

    "I don't know what happened",

    "asdfgh random text"
]


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n")
print("=" * 70)
print("CONFIDENCE TEST")
print("=" * 70)


for ticket in tickets:

    (
        category,
        category_confidence,
        intent,
        intent_confidence
    ) = predict_ticket(ticket)


    print("\nTicket:", ticket)

    print(
        f"Category: {category} "
        f"({category_confidence:.2%})"
    )

    print(
        f"Intent: {intent} "
        f"({intent_confidence:.2%})"
    )


    # ------------------------------------------------
    # HUMAN REVIEW DECISION
    # ------------------------------------------------

    if (
        category_confidence < 0.70
        or intent_confidence < 0.70
    ):

        print(
            "Decision: ⚠️ HUMAN REVIEW"
        )

    else:

        print(
            "Decision: ✅ CONTINUE"
        )