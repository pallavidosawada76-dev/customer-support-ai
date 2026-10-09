from datasets import load_dataset
import pandas as pd


# Load dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = pd.DataFrame(dataset)


# --------------------------------------------------
# Inspect specific intents
# --------------------------------------------------

intents_to_check = [
    "payment_issue",
    "check_payment_methods",
    "track_order",
    "cancel_order",
    "change_shipping_address"
]


for intent in intents_to_check:

    print("\n")
    print("=" * 70)
    print(f"INTENT: {intent}")
    print("=" * 70)

    examples = df[df["intent"] == intent]["instruction"]

    print(f"Total examples: {len(examples)}")

    print("\nSample examples:\n")

    for example in examples.sample(
        min(15, len(examples)),
        random_state=42
    ):
        print("-", example)
        