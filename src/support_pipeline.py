from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


# ============================================================
# 3. TF-IDF FOR CATEGORY MODEL
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================================
# 4. TRAIN CATEGORY MODEL
# ============================================================

print("\nTraining category model...")

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train_tfidf,
    y_category_train
)


# ============================================================
# 5. EVALUATE CATEGORY MODEL
# ============================================================

category_predictions = category_model.predict(
    X_test_tfidf
)

category_accuracy = accuracy_score(
    y_category_test,
    category_predictions
)

print("\n========== CATEGORY MODEL ==========")

print("Accuracy:", category_accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_category_test,
        category_predictions
    )
)


# ============================================================
# 6. BUILD CATEGORY → INTENT MAPPING
# ============================================================

category_intents = {}

for category, intent in zip(
    y_category_train,
    y_intent_train
):

    if category not in category_intents:
        category_intents[category] = set()

    category_intents[category].add(intent)


print("\n========== CATEGORY → INTENT MAPPING ==========")

for category, intents in sorted(category_intents.items()):

    print("\n", category)

    for intent in sorted(intents):
        print("  →", intent)


# ============================================================
# 7. TRAIN CATEGORY-SPECIFIC INTENT MODELS
# ============================================================

print("\n========== TRAINING INTENT MODELS ==========")

intent_models = {}

for category in sorted(category_intents):

    # --------------------------------------------------------
    # Get training examples belonging to this category
    # --------------------------------------------------------

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


    # Find unique intents
    unique_intents = set(category_y)


    # --------------------------------------------------------
    # CASE 1: Category has only ONE intent
    # --------------------------------------------------------

    if len(unique_intents) == 1:

        only_intent = list(unique_intents)[0]

        intent_models[category] = {
            "type": "single",
            "intent": only_intent
        }

        print(
            f"{category}: "
            f"{len(category_X)} training examples, "
            f"1 intent → {only_intent}"
        )

        continue


    # --------------------------------------------------------
    # CASE 2: Category has MULTIPLE intents
    # --------------------------------------------------------

    category_vectorizer = TfidfVectorizer()

    category_X_tfidf = category_vectorizer.fit_transform(
        category_X
    )


    # Train intent classifier
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        category_X_tfidf,
        category_y
    )


    # Save model + vectorizer
    intent_models[category] = {
        "type": "model",
        "model": model,
        "vectorizer": category_vectorizer
    }


    print(
        f"{category}: "
        f"{len(category_X)} training examples, "
        f"{len(unique_intents)} intents"
    )


# ============================================================
# 8. PROCESS A NEW SUPPORT TICKET
# ============================================================

def process_ticket(ticket):

    # --------------------------------------------------------
    # STEP 1: Convert ticket to TF-IDF
    # --------------------------------------------------------

    ticket_tfidf = vectorizer.transform(
        [ticket]
    )


    # --------------------------------------------------------
    # STEP 2: Predict CATEGORY
    # --------------------------------------------------------

    predicted_category = category_model.predict(
        ticket_tfidf
    )[0]


    # --------------------------------------------------------
    # STEP 3: Get correct intent model
    # --------------------------------------------------------

    selected_intent_model = intent_models[
        predicted_category
    ]


    # --------------------------------------------------------
    # STEP 4: Predict INTENT
    # --------------------------------------------------------

    if selected_intent_model["type"] == "single":

        # Category has only one possible intent
        predicted_intent = selected_intent_model[
            "intent"
        ]

    else:

        # Category has multiple possible intents
        intent_model = selected_intent_model[
            "model"
        ]

        intent_vectorizer = selected_intent_model[
            "vectorizer"
        ]


        # Transform ticket using the category-specific
        # TF-IDF vectorizer
        ticket_intent_tfidf = intent_vectorizer.transform(
            [ticket]
        )


        predicted_intent = intent_model.predict(
            ticket_intent_tfidf
        )[0]


    # --------------------------------------------------------
    # STEP 5: ROUTE TICKET
    # --------------------------------------------------------

    assigned_team = route_ticket(
        predicted_category
    )


    return (
        predicted_category,
        predicted_intent,
        assigned_team
    )


# ============================================================
# 9. TEST THE COMPLETE PIPELINE
# ============================================================

test_tickets = [

    "I forgot my password and cannot access my account",

    "My payment failed and I was charged for the transaction",

    "I want to change the address for my delivery",

    "Where is my order? It has not arrived yet",

    "I need help getting a refund",

    "What payment methods do you accept?",

    "I want to cancel my order"

]


print("\n\n========== SUPPORT TICKET PIPELINE ==========")


for ticket in test_tickets:

    category, intent, team = process_ticket(
        ticket
    )

    print("\nTicket:", ticket)

    print("Category:", category)

    print("Intent:", intent)

    print("Assigned Team:", team)