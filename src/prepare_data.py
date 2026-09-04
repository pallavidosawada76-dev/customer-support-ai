from datasets import load_dataset

# Load dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

data = dataset["train"]

# Create X and y
X = [row["instruction"] for row in data]
y = [row["category"] for row in data]

print("Number of inputs:", len(X))
print("Number of labels:", len(y))

print("\nFirst input:")
print(X[0])

print("\nFirst label:")
print(y[0])



from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

data = dataset["train"]

# Create X and y
X = [row["instruction"] for row in data]
y = [row["category"] for row in data]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Total examples:", len(X))
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

print("\nExample from training data:")
print(X_train[0])
print("Label:", y_train[0])

print("\nExample from testing data:")
print(X_test[0])
print("Label:", y_test[0])