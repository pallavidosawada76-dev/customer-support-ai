from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. LOAD ORIGINAL DATA
# ==================================================

print("Loading original dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = pd.DataFrame(dataset)

X = df["instruction"]
y_category = df["category"]
y_intent = df["intent"]


# ==================================================
# 2. CREATE FIXED ORIGINAL TRAIN / TEST SPLIT
# ==================================================

(
    X_train_original,
    X_test,
    y_cat_train_original,
    y_cat_test,
    y_intent_train_original,
    y_intent_test
) = train_test_split(
    X,
    y_category,
    y_intent,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

print("\nOriginal training examples:", len(X_train_original))
print("Original test examples:", len(X_test))


# ==================================================
# 3. LOAD AUGMENTED DATA
# ==================================================

augmented_df = pd.read_csv(
    "data/augmented_dataset.csv"
)

# Identify only the NEW examples
new_examples = augmented_df.iloc[len(df):]

print("\nNew augmented examples:", len(new_examples))


# ==================================================
# 4. ADD AUGMENTED EXAMPLES ONLY TO TRAINING DATA
# ==================================================

X_train = pd.concat(
    [
        X_train_original.reset_index(drop=True),
        new_examples["instruction"].reset_index(drop=True)
    ],
    ignore_index=True
)

y_cat_train = pd.concat(
    [
        y_cat_train_original.reset_index(drop=True),
        new_examples["category"].reset_index(drop=True)
    ],
    ignore_index=True
)

y_intent_train = pd.concat(
    [
        y_intent_train_original.reset_index(drop=True),
        new_examples["intent"].reset_index(drop=True)
    ],
    ignore_index=True
)

print("\nFinal training examples:", len(X_train))
print("Final test examples:", len(X_test))


# ==================================================
# 5. TRAIN CATEGORY MODEL
# ==================================================

print("\nTraining category model...")

category_vectorizer = TfidfVectorizer()

X_train_category = category_vectorizer.fit_transform(X_train)
X_test_category = category_vectorizer.transform(X_test)

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train_category,
    y_cat_train
)


# ==================================================
# 6. BUILD CATEGORY → INTENT MAPPING
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
# 7. TRAIN CATEGORY-SPECIFIC INTENT MODELS
# ==================================================

intent_models = {}
intent_vectorizers = {}

print("\nTraining intent models...")

for category, intents in category_intents.items():

    # Only one intent → no classifier needed
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
# 8. PREDICT CATEGORY
# ==================================================

predicted_categories = category_model.predict(
    X_test_category
)


# ==================================================
# 9. PREDICT INTENT
# ==================================================

predicted_intents = []

for text, predicted_category in zip(
    X_test,
    predicted_categories
):

    # Category has only one possible intent
    if predicted_category not in intent_models:

        intent = list(
            category_intents[predicted_category]
        )[0]

        predicted_intents.append(intent)

    else:

        vectorizer = intent_vectorizers[
            predicted_category
        ]

        model = intent_models[
            predicted_category
        ]

        text_vector = vectorizer.transform(
            [text]
        )

        intent = model.predict(
            text_vector
        )[0]

        predicted_intents.append(intent)


# ==================================================
# 10. EVALUATE CATEGORY
# ==================================================

print("\n")
print("=" * 70)
print("CLEAN CATEGORY EVALUATION")
print("=" * 70)

category_accuracy = accuracy_score(
    y_cat_test,
    predicted_categories
)

print(
    f"\nCategory Accuracy: "
    f"{category_accuracy:.4f}"
)

print("\nCategory Classification Report:")

print(
    classification_report(
        y_cat_test,
        predicted_categories
    )
)


# ==================================================
# 11. CATEGORY CONFUSION MATRIX
# ==================================================

print("\nGenerating Category Confusion Matrix...")

category_labels = sorted(
    y_cat_test.unique()
)

category_cm = confusion_matrix(
    y_cat_test,
    predicted_categories,
    labels=category_labels
)

plt.figure(figsize=(11, 9))

sns.heatmap(
    category_cm,
    annot=True,
    fmt="d",
    xticklabels=category_labels,
    yticklabels=category_labels
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Category Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "category_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Saved: category_confusion_matrix.png"
)

plt.show()


# ==================================================
# 12. EVALUATE INTENT
# ==================================================

print("\n")
print("=" * 70)
print("CLEAN INTENT EVALUATION")
print("=" * 70)

intent_accuracy = accuracy_score(
    y_intent_test,
    predicted_intents
)

print(
    f"\nIntent Accuracy: "
    f"{intent_accuracy:.4f}"
)

print("\nIntent Classification Report:")

print(
    classification_report(
        y_intent_test,
        predicted_intents
    )
)


# ==================================================
# 13. INTENT CONFUSION MATRIX
# ==================================================

print("\nGenerating Intent Confusion Matrix...")

intent_labels = sorted(
    y_intent_test.unique()
)

intent_cm = confusion_matrix(
    y_intent_test,
    predicted_intents,
    labels=intent_labels
)

plt.figure(figsize=(18, 16))

sns.heatmap(
    intent_cm,
    annot=True,
    fmt="d",
    xticklabels=intent_labels,
    yticklabels=intent_labels
)

plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("Intent Confusion Matrix")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "intent_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Saved: intent_confusion_matrix.png"
)

plt.show()


# ==================================================
# 14. REAL-WORLD TEST CASES
# ==================================================

print("\n")
print("=" * 70)
print("REAL-WORLD TEST CASES")
print("=" * 70)


real_world_tickets = [

    "My card was charged twice",

    "I was charged twice for the same transaction",

    "There is a duplicate charge on my card",

    "There is an extra charge for my purchase",

    "My payment failed",

    "I cannot complete my payment",

    "Where is my order?",

    "I want to track my order",

    "I want to cancel my order",

    "I want to change my order",

    "I forgot my password",

    "I want a refund",

    "I want to change my shipping address",

    "Where is my delivery?"
]


for ticket in real_world_tickets:

    category_vector = category_vectorizer.transform(
        [ticket]
    )

    category = category_model.predict(
        category_vector
    )[0]


    if category not in intent_models:

        intent = list(
            category_intents[category]
        )[0]

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

        intent = model.predict(
            ticket_vector
        )[0]


    print("\nTicket:", ticket)

    print("Category:", category)

    print("Intent:", intent)


# ==================================================
# 15. FINAL MESSAGE
# ==================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("1. category_confusion_matrix.png")

print("2. intent_confusion_matrix.png")

print("\nCategory Accuracy:",
      f"{category_accuracy:.4f}")

print("Intent Accuracy:",
      f"{intent_accuracy:.4f}")

print("\nAll evaluation tasks completed successfully!")