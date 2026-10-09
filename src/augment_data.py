import pandas as pd
from datasets import load_dataset


# --------------------------------------------------
# 1. LOAD ORIGINAL DATA
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = pd.DataFrame(dataset)


# --------------------------------------------------
# 2. NEW REAL-WORLD PAYMENT EXAMPLES
# --------------------------------------------------

new_examples = [
    "My card was charged twice",
    "I was charged twice for the same transaction",
    "I was billed twice for the same purchase",
    "There is a duplicate charge on my card",
    "Why was I charged twice?",
    "I see two charges for the same order",
    "The same payment was charged twice",
    "I got charged two times for one purchase",
    "My account shows a duplicate payment",
    "I was charged more than once for the same order",
    "There are two identical charges on my card",
    "The payment appears twice on my statement",
    "I paid once but I was charged twice",
    "I made one payment but see two charges",
    "Why do I have two charges for one transaction",
    "My card has been charged twice for one purchase",
    "I see a duplicate transaction",
    "There is an extra charge for my purchase",
    "The same transaction appears twice",
    "I was charged twice by mistake",
    "I think I was billed twice",
    "I have two charges for the same item",
    "I was charged two times for the same order",
    "There seems to be a duplicate charge",
    "My payment was processed twice",
]


# --------------------------------------------------
# 3. CREATE DATAFRAME
# --------------------------------------------------

augmented = pd.DataFrame({
    "flags": [None] * len(new_examples),
    "instruction": new_examples,
    "category": ["PAYMENT"] * len(new_examples),
    "intent": ["payment_issue"] * len(new_examples),
    "response": [""] * len(new_examples)
})


# --------------------------------------------------
# 4. COMBINE DATA
# --------------------------------------------------

df_augmented = pd.concat(
    [df, augmented],
    ignore_index=True
)


# --------------------------------------------------
# 5. SAVE
# --------------------------------------------------

output_path = "data/augmented_dataset.csv"

df_augmented.to_csv(
    output_path,
    index=False
)


print("\nOriginal dataset:", len(df))
print("New examples:", len(augmented))
print("Augmented dataset:", len(df_augmented))

print("\nSaved to:")
print(output_path)