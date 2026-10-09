# ============================================
# CUSTOMER SUPPORT AI - DATA PREPARATION
# ============================================

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# 1. LOAD DATASET
# ============================================

data = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

print("Number of inputs:", len(data))
print("Number of labels:", len(data))


# ============================================
# 2. CREATE INPUTS AND TARGETS
# ============================================

X = [row["instruction"] for row in data]

# Category is our first target
y_category = [row["category"] for row in data]

# Intent is our second target
y_intent = [row["intent"] for row in data]


print("\nFirst input:")
print(X[0])

print("\nFirst category:")
print(y_category[0])

print("\nFirst intent:")
print(y_intent[0])


# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================

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


print("\n========== TRAIN / TEST SPLIT ==========")

print("Total examples:", len(X))
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

print("\nExample from training data:")
print(X_train[0])
print("Category:", y_category_train[0])
print("Intent:", y_intent_train[0])

print("\nExample from testing data:")
print(X_test[0])
print("Category:", y_category_test[0])
print("Intent:", y_intent_test[0])


# ============================================
# 4. TF-IDF
# ============================================

vectorizer = TfidfVectorizer()

# Learn vocabulary from training data
# and transform training text
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using the SAME vocabulary
X_test_tfidf = vectorizer.transform(X_test)


print("\n========== TF-IDF ==========")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================
# 5. CATEGORY CLASSIFIER
# ============================================

category_model = LogisticRegression(max_iter=1000)

# Train category model
category_model.fit(
    X_train_tfidf,
    y_category_train
)


print("\n========== CATEGORY MODEL ==========")
print("Category model trained successfully!")


# ============================================
# 6. CATEGORY PREDICTIONS
# ============================================

y_category_pred = category_model.predict(X_test_tfidf)


print("\n========== CATEGORY PREDICTIONS ==========")

print("First 10 actual categories:")
print(y_category_test[:10])

print("\nFirst 10 predicted categories:")
print(y_category_pred[:10])


# ============================================
# 7. CATEGORY EVALUATION
# ============================================

category_accuracy = accuracy_score(
    y_category_test,
    y_category_pred
)

print("\n========== CATEGORY MODEL EVALUATION ==========")

print("Category Accuracy:", category_accuracy)


print("\n========== CATEGORY CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_category_test,
        y_category_pred
    )
)


# ============================================
# 8. CATEGORY CONFUSION MATRIX
# ============================================

category_cm = confusion_matrix(
    y_category_test,
    y_category_pred,
    labels=category_model.classes_
)


plt.figure(figsize=(10, 8))

sns.heatmap(
    category_cm,
    annot=True,
    fmt="d",
    xticklabels=category_model.classes_,
    yticklabels=category_model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Category Confusion Matrix")

plt.show()


# ============================================
# 9. INTENT CLASSIFIER
# ============================================

intent_model = LogisticRegression(max_iter=1000)

# Train intent model
intent_model.fit(
    X_train_tfidf,
    y_intent_train
)


print("\n========== INTENT MODEL ==========")
print("Intent model trained successfully!")


# ============================================
# 10. INTENT PREDICTIONS
# ============================================

y_intent_pred = intent_model.predict(X_test_tfidf)


print("\n========== INTENT PREDICTIONS ==========")

print("First 10 actual intents:")
print(y_intent_test[:10])

print("\nFirst 10 predicted intents:")
print(y_intent_pred[:10])


# ============================================
# 11. INTENT EVALUATION
# ============================================

intent_accuracy = accuracy_score(
    y_intent_test,
    y_intent_pred
)

print("\n========== INTENT MODEL EVALUATION ==========")

print("Intent Accuracy:", intent_accuracy)


print("\n========== INTENT CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_intent_test,
        y_intent_pred
    )
)


# ============================================
# 12. INTENT CONFUSION MATRIX
# ============================================

intent_cm = confusion_matrix(
    y_intent_test,
    y_intent_pred,
    labels=intent_model.classes_
)


plt.figure(figsize=(14, 10))

sns.heatmap(
    intent_cm,
    annot=True,
    fmt="d",
    xticklabels=intent_model.classes_,
    yticklabels=intent_model.classes_
)

plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("Intent Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================
# 13. TEST WITH NEW CUSTOMER TICKETS
# ============================================

new_tickets = [
    "I forgot my password and cannot access my account",
    "Where is my order? It was supposed to arrive yesterday",
    "I want to know how I can get a refund",
    "What payment methods do you accept?",
    "Can I change my shipping address?"
]


# Convert new tickets using the SAME vectorizer
new_tickets_tfidf = vectorizer.transform(new_tickets)


# Predict category
new_category_predictions = category_model.predict(
    new_tickets_tfidf
)


# Predict intent
new_intent_predictions = intent_model.predict(
    new_tickets_tfidf
)


print("\n========== NEW TICKET PREDICTIONS ==========")


for ticket, category, intent in zip(
    new_tickets,
    new_category_predictions,
    new_intent_predictions
):

    print("\nTicket:", ticket)
    print("Predicted category:", category)
    print("Predicted intent:", intent)