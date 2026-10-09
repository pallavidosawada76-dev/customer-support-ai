from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("data/augmented_dataset.csv")

X = df["instruction"]
y_category = df["category"]
y_intent = df["intent"]


# --------------------------------------------------
# 2. SAME TRAIN/TEST SPLIT AS BEFORE
# --------------------------------------------------

X_train, X_test, y_cat_train, y_cat_test, y_int_train, y_int_test = train_test_split(
    X,
    y_category,
    y_intent,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)


# --------------------------------------------------
# 3. TRAIN CATEGORY MODEL
# --------------------------------------------------

print("\nTraining category model...")

category_vectorizer = TfidfVectorizer()

X_train_cat = category_vectorizer.fit_transform(X_train)
X_test_cat = category_vectorizer.transform(X_test)

category_model = LogisticRegression(max_iter=1000)

category_model.fit(X_train_cat, y_cat_train)


# --------------------------------------------------
# 4. BUILD CATEGORY → INTENT MAPPING
# --------------------------------------------------

category_intents = {}

for category, intent in zip(y_cat_train, y_int_train):

    if category not in category_intents:
        category_intents[category] = set()

    category_intents[category].add(intent)


# --------------------------------------------------
# 5. TRAIN INTENT MODELS
# --------------------------------------------------

intent_models = {}
intent_vectorizers = {}

print("\nTraining intent models...")

for category, intents in category_intents.items():

    if len(intents) == 1:
        continue

    mask = y_cat_train == category

    category_text = X_train[mask]
    category_intent = y_int_train[mask]

    vectorizer = TfidfVectorizer()

    X_category = vectorizer.fit_transform(category_text)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_category, category_intent)

    intent_vectorizers[category] = vectorizer
    intent_models[category] = model


# --------------------------------------------------
# 6. PREDICT INTENTS ON TEST SET
# --------------------------------------------------

predicted_categories = category_model.predict(X_test_cat)

predicted_intents = []

for text, predicted_category in zip(X_test, predicted_categories):

    # Single-intent category
    if predicted_category not in intent_models:

        single_intent = list(category_intents[predicted_category])[0]
        predicted_intents.append(single_intent)

    else:

        vectorizer = intent_vectorizers[predicted_category]
        model = intent_models[predicted_category]

        text_vector = vectorizer.transform([text])

        predicted_intent = model.predict(text_vector)[0]

        predicted_intents.append(predicted_intent)


# --------------------------------------------------
# 7. CATEGORY EVALUATION
# --------------------------------------------------

print("\n")
print("=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

category_accuracy = accuracy_score(
    y_cat_test,
    predicted_categories
)

print(f"\nCategory Accuracy: {category_accuracy:.4f}")

print("\nCategory Classification Report:")

print(
    classification_report(
        y_cat_test,
        predicted_categories
    )
)


# --------------------------------------------------
# 8. INTENT EVALUATION
# --------------------------------------------------

print("\n")
print("=" * 70)
print("INTENT PERFORMANCE")
print("=" * 70)

intent_accuracy = accuracy_score(
    y_int_test,
    predicted_intents
)

print(f"\nOverall Intent Accuracy: {intent_accuracy:.4f}")

print("\nIntent Classification Report:")

print(
    classification_report(
        y_int_test,
        predicted_intents
    )
)


# --------------------------------------------------
# 9. SHOW WRONG PREDICTIONS
# --------------------------------------------------

print("\n")
print("=" * 70)
print("SAMPLE WRONG PREDICTIONS")
print("=" * 70)

wrong_count = 0

for text, actual_category, predicted_category, actual_intent, predicted_intent in zip(
    X_test,
    y_cat_test,
    predicted_categories,
    y_int_test,
    predicted_intents
):

    if actual_intent != predicted_intent:

        print("\nTicket:", text)
        print("Actual Category:", actual_category)
        print("Predicted Category:", predicted_category)
        print("Actual Intent:", actual_intent)
        print("Predicted Intent:", predicted_intent)

        wrong_count += 1

        if wrong_count >= 20:
            break