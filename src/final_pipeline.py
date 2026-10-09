import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CUSTOMER SUPPORT AI - FINAL PIPELINE
# ============================================================
#
# Pipeline:
#
# Customer Ticket
#       ↓
# Category Classification
#       ↓
# Intent Classification
#       ↓
# Team Routing
#       ↓
# RAG Retrieval
#       ↓
# Draft Response
#       ↓
# Human Approval
#
# ============================================================


print("\n" + "=" * 70)
print("CUSTOMER SUPPORT AI - FINAL PIPELINE")
print("=" * 70)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_PATH = "data/augmented_dataset.csv"

KB_PATH = "data/knowledge_base.csv"

EMBEDDING_PATH = (
    "data/embeddings/ticket_embeddings.npy"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("1. LOADING DATASET")
print("-" * 70)


data = pd.read_csv(DATA_PATH)

print(
    "Dataset size:",
    len(data)
)


# Make sure important columns do not contain NaN

data["instruction"] = (
    data["instruction"]
    .fillna("")
    .astype(str)
)

data["category"] = (
    data["category"]
    .fillna("")
    .astype(str)
)

data["intent"] = (
    data["intent"]
    .fillna("")
    .astype(str)
)


# ============================================================
# 3. PREPARE FEATURES AND LABELS
# ============================================================

X = data["instruction"]

y_category = data["category"]

y_intent = data["intent"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("2. SPLITTING DATA")
print("-" * 70)


(
    X_train,
    X_test,
    y_cat_train,
    y_cat_test,
    y_int_train,
    y_int_test
) = train_test_split(

    X,
    y_category,
    y_intent,

    test_size=0.20,

    random_state=42,

    stratify=y_category
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 5. CATEGORY CLASSIFIER
# ============================================================

print("\n" + "-" * 70)
print("3. TRAINING CATEGORY CLASSIFIER")
print("-" * 70)


category_vectorizer = TfidfVectorizer(

    lowercase=True,

    max_features=10000,

    ngram_range=(1, 2)
)


X_cat_train = (
    category_vectorizer.fit_transform(
        X_train
    )
)


X_cat_test = (
    category_vectorizer.transform(
        X_test
    )
)


category_model = LogisticRegression(

    max_iter=1000
)


category_model.fit(

    X_cat_train,

    y_cat_train
)


# Evaluate category classifier

cat_predictions = (
    category_model.predict(
        X_cat_test
    )
)


category_accuracy = (
    accuracy_score(
        y_cat_test,
        cat_predictions
    )
)


print(
    "\nCategory Accuracy:",
    round(
        category_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 6. CREATE CATEGORY -> INTENT MAPPING
# ============================================================

print("\n" + "-" * 70)
print("4. CREATING CATEGORY -> INTENT MAPPING")
print("-" * 70)


CATEGORY_INTENTS = {}


for category in sorted(
    data["category"].unique()
):

    intents = sorted(
        data[
            data["category"] == category
        ]["intent"].unique()
    )

    CATEGORY_INTENTS[
        category
    ] = intents


for category, intents in (
    CATEGORY_INTENTS.items()
):

    print(
        f"{category}: "
        f"{', '.join(intents)}"
    )


# ============================================================
# 7. TRAIN CATEGORY-SPECIFIC INTENT MODELS
# ============================================================

print("\n" + "-" * 70)
print("5. TRAINING HIERARCHICAL INTENT MODELS")
print("-" * 70)


intent_models = {}

intent_vectorizers = {}


for category, intents in (
    CATEGORY_INTENTS.items()
):

    category_data = data[
        data["category"] == category
    ]


    category_text = (
        category_data["instruction"]
    )

    category_intent = (
        category_data["intent"]
    )


    # --------------------------------------------------------
    # SINGLE INTENT CATEGORY
    # --------------------------------------------------------

    if len(intents) == 1:

        intent_models[
            category
        ] = {
            "single_intent":
                intents[0]
        }


        print(
            f"{category}: "
            f"{len(category_data)} examples "
            f"-> single intent "
            f"-> {intents[0]}"
        )


        continue


    # --------------------------------------------------------
    # MULTIPLE INTENTS
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        lowercase=True,

        max_features=10000,

        ngram_range=(1, 2)
    )


    X_intent = (
        vectorizer.fit_transform(
            category_text
        )
    )


    model = LogisticRegression(

        max_iter=1000
    )


    model.fit(

        X_intent,

        category_intent
    )


    intent_vectorizers[
        category
    ] = vectorizer


    intent_models[
        category
    ] = model


    print(
        f"{category}: "
        f"{len(category_data)} examples "
        f"-> {len(intents)} intents"
    )


# ============================================================
# 8. LOAD KNOWLEDGE BASE
# ============================================================

print("\n" + "-" * 70)
print("6. LOADING KNOWLEDGE BASE")
print("-" * 70)


knowledge_base = pd.read_csv(
    KB_PATH
)


# Handle missing values

knowledge_base["instruction"] = (
    knowledge_base["instruction"]
    .fillna("")
    .astype(str)
)


knowledge_base["category"] = (
    knowledge_base["category"]
    .fillna("")
    .astype(str)
)


knowledge_base["intent"] = (
    knowledge_base["intent"]
    .fillna("")
    .astype(str)
)


print(
    "Knowledge base size:",
    len(knowledge_base)
)


# ============================================================
# 9. LOAD EMBEDDINGS
# ============================================================

print("\n" + "-" * 70)
print("7. LOADING EMBEDDINGS")
print("-" * 70)


embeddings = np.load(
    EMBEDDING_PATH
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# 10. CHECK DATA / EMBEDDING SIZE
# ============================================================

if len(knowledge_base) != len(embeddings):

    raise ValueError(
        "\nKnowledge base and embeddings "
        "have different sizes!\n"
        f"Knowledge base: {len(knowledge_base)}\n"
        f"Embeddings: {len(embeddings)}\n"
        "Please recreate the embeddings."
    )


# ============================================================
# 11. LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer model...")


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded successfully."
)


# ============================================================
# 12. ROUTING MAP
# ============================================================

ROUTING_MAP = {

    "ACCOUNT":
        "Account Support",

    "PAYMENT":
        "Payments Team",

    "SHIPPING":
        "Shipping Team",

    "DELIVERY":
        "Delivery Team",

    "REFUND":
        "Refunds Team",

    "ORDER":
        "Order Support",

    "INVOICE":
        "Billing Team",

    "SUBSCRIPTION":
        "Subscription Team",

    "CANCEL":
        "Cancellation Team",

    "CONTACT":
        "Customer Support",

    "FEEDBACK":
        "Customer Feedback Team"
}


def route_ticket(category):

    return ROUTING_MAP.get(

        category,

        "General Support"
    )


# ============================================================
# 13. CATEGORY PREDICTION
# ============================================================

def predict_category(ticket):

    ticket_vector = (
        category_vectorizer.transform(
            [ticket]
        )
    )


    probabilities = (
        category_model.predict_proba(
            ticket_vector
        )[0]
    )


    predicted_index = (
        np.argmax(
            probabilities
        )
    )


    category = (
        category_model.classes_[
            predicted_index
        ]
    )


    confidence = (
        probabilities[
            predicted_index
        ]
    )


    return category, confidence


# ============================================================
# 14. INTENT PREDICTION
# ============================================================

def predict_intent(
    ticket,
    category
):

    model_info = (
        intent_models[
            category
        ]
    )


    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Some categories contain only one intent.
    # In that case we stored a dictionary.
    #
    # Other categories contain multiple intents.
    # In that case we stored a LogisticRegression model.
    #
    # Therefore we check the TYPE first.
    # --------------------------------------------------------

    if (
        isinstance(
            model_info,
            dict
        )
        and
        "single_intent" in model_info
    ):

        return (

            model_info[
                "single_intent"
            ],

            1.0
        )


    # --------------------------------------------------------
    # MULTI-INTENT CATEGORY
    # --------------------------------------------------------

    vectorizer = (
        intent_vectorizers[
            category
        ]
    )


    model = model_info


    ticket_vector = (
        vectorizer.transform(
            [ticket]
        )
    )


    probabilities = (
        model.predict_proba(
            ticket_vector
        )[0]
    )


    predicted_index = (
        np.argmax(
            probabilities
        )
    )


    intent = (
        model.classes_[
            predicted_index
        ]
    )


    confidence = (
        probabilities[
            predicted_index
        ]
    )


    return intent, confidence


# ============================================================
# 15. RAG RETRIEVAL
# ============================================================

def retrieve_similar_tickets(

    ticket,

    category,

    intent,

    top_k=3

):

    # --------------------------------------------------------
    # Convert incoming ticket into embedding
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            [ticket]
        )
    )


    # --------------------------------------------------------
    # First try category + intent
    # --------------------------------------------------------

    filtered_indices = (
        knowledge_base.index[
            (
                knowledge_base[
                    "category"
                ] == category
            )
            &
            (
                knowledge_base[
                    "intent"
                ] == intent
            )
        ].tolist()
    )


    # --------------------------------------------------------
    # If there are not enough examples,
    # fallback to category only.
    # --------------------------------------------------------

    if len(
        filtered_indices
    ) < 3:

        filtered_indices = (
            knowledge_base.index[
                knowledge_base[
                    "category"
                ] == category
            ].tolist()
        )


    # --------------------------------------------------------
    # Final safety fallback
    # --------------------------------------------------------

    if len(
        filtered_indices
    ) == 0:

        filtered_indices = list(
            range(
                len(
                    knowledge_base
                )
            )
        )


    # --------------------------------------------------------
    # Get embeddings of filtered tickets
    # --------------------------------------------------------

    filtered_embeddings = (
        embeddings[
            filtered_indices
        ]
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = (
        cosine_similarity(
            query_embedding,
            filtered_embeddings
        )[0]
    )


    # --------------------------------------------------------
    # Get highest similarity results
    # --------------------------------------------------------

    top_positions = (
        np.argsort(
            similarities
        )[::-1][:top_k]
    )


    results = []


    for position in (
        top_positions
    ):

        original_index = (
            filtered_indices[
                position
            ]
        )


        row = (
            knowledge_base.iloc[
                original_index
            ]
        )


        results.append({

            "instruction":
                row["instruction"],

            "category":
                row["category"],

            "intent":
                row["intent"],

            "response":
                row["response"],

            "similarity":
                float(
                    similarities[
                        position
                    ]
                )
        })


    return results


# ============================================================
# 16. GENERATE DRAFT RESPONSE
# ============================================================

def generate_draft_response(

    ticket,

    result

):

    # --------------------------------------------------------
    # Get retrieved response
    # --------------------------------------------------------

    response = result.get(
        "response"
    )


    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Blank responses in CSV become NaN.
    # pandas represents NaN as a float.
    #
    # Calling .replace() on NaN causes:
    #
    # AttributeError:
    # 'float' object has no attribute 'replace'
    #
    # --------------------------------------------------------

    if (
        pd.isna(response)
        or
        not isinstance(
            response,
            str
        )
    ):

        response = ""


    # --------------------------------------------------------
    # Replace common placeholders
    # --------------------------------------------------------

    response = response.replace(

        "{{Login Page URL}}",

        "the login page"
    )


    response = response.replace(

        "{{Forgot Password}}",

        "Forgot Password"
    )


    response = response.replace(

        "{{Order Number}}",

        "your order number"
    )


    response = response.replace(

        "{{Customer Support Phone Number}}",

        "our support phone number"
    )


    response = response.replace(

        "{{Website URL}}",

        "our website"
    )


    response = response.replace(

        "{{Customer Support Hours}}",

        "our support hours"
    )


    # ========================================================
    # FALLBACK RESPONSES
    # ========================================================

    if not response.strip():


        # ----------------------------------------------------
        # PAYMENT ISSUE
        # ----------------------------------------------------

        if (
            result["intent"]
            == "payment_issue"
        ):

            response = (

                "We're sorry to hear that you "
                "experienced a problem with your payment. "
                "We understand that seeing a duplicate "
                "charge can be concerning. "
                "Please provide the transaction details "
                "so our Payments Team can investigate "
                "the charge and assist you with the issue."
            )


        # ----------------------------------------------------
        # TRACK ORDER
        # ----------------------------------------------------

        elif (
            result["intent"]
            == "track_order"
        ):

            response = (

                "We'd be happy to help you locate "
                "your order. Please provide your "
                "order number or the email address "
                "associated with the purchase so our "
                "Order Support team can check its status."
            )


        # ----------------------------------------------------
        # RECOVER PASSWORD
        # ----------------------------------------------------

        elif (
            result["intent"]
            == "recover_password"
        ):

            response = (

                "To recover your account, please use "
                "the Forgot Password option on the "
                "login page. You will receive instructions "
                "to reset your password through your "
                "registered email address."
            )


        # ----------------------------------------------------
        # GENERIC FALLBACK
        # ----------------------------------------------------

        else:

            response = (

                "Thank you for contacting our support team. "
                "We have received your request and our "
                "support team will assist you with the issue."
            )


    # ========================================================
    # FINAL DRAFT
    # ========================================================

    draft = (

        "Hello,\n\n"

        "Thank you for contacting our support team. "
        "We understand your issue and are happy to help.\n\n"

        + response

        +

        "\n\n"

        "If the issue continues, please provide any "
        "additional details so our support team can "
        "investigate further.\n\n"

        "Best regards,\n"
        "Customer Support Team"
    )


    return draft


# ============================================================
# 17. CONFIDENCE CHECK
# ============================================================

def confidence_check(

    category_confidence,

    intent_confidence,

    threshold=0.50

):

    if (

        category_confidence
        >= threshold

        and

        intent_confidence
        >= threshold

    ):

        return True


    return False


# ============================================================
# 18. PROCESS COMPLETE TICKET
# ============================================================

def process_ticket(ticket):

    print("\n")
    print("=" * 70)
    print("NEW SUPPORT TICKET")
    print("=" * 70)


    print("\nTicket:")

    print(ticket)


    # ========================================================
    # CATEGORY
    # ========================================================

    category, category_confidence = (
        predict_category(
            ticket
        )
    )


    print("\nCategory:")

    print(category)


    print(

        "Category confidence:",

        round(
            category_confidence * 100,
            2
        ),

        "%"
    )


    # ========================================================
    # INTENT
    # ========================================================

    intent, intent_confidence = (
        predict_intent(

            ticket,

            category
        )
    )


    print("\nIntent:")

    print(intent)


    print(

        "Intent confidence:",

        round(
            intent_confidence * 100,
            2
        ),

        "%"
    )


    # ========================================================
    # ROUTING
    # ========================================================

    team = route_ticket(
        category
    )


    print("\nAssigned Team:")

    print(team)


    # ========================================================
    # CONFIDENCE CHECK
    # ========================================================

    confident = confidence_check(

        category_confidence,

        intent_confidence,

        threshold=0.50
    )


    # ========================================================
    # HUMAN REVIEW IF LOW CONFIDENCE
    # ========================================================

    if not confident:

        print("\n" + "!" * 70)

        print(
            "HUMAN REVIEW REQUIRED"
        )

        print("!" * 70)


        print(

            "\nThe AI is not sufficiently "
            "confident about this ticket."
        )


        return {

            "ticket":
                ticket,

            "category":
                category,

            "category_confidence":
                category_confidence,

            "intent":
                intent,

            "intent_confidence":
                intent_confidence,

            "team":
                team,

            "human_review":
                True,

            "rag_results":
                [],

            "draft_response":
                None
        }


    # ========================================================
    # RAG
    # ========================================================

    print(
        "\nRetrieving similar resolved tickets..."
    )


    rag_results = (
        retrieve_similar_tickets(

            ticket,

            category,

            intent,

            top_k=3
        )
    )


    print(
        "\nTop Similar Tickets:"
    )


    for i, rag_result in enumerate(

        rag_results,

        start=1

    ):

        print(

            f"\n{i}. "
            f"Similarity: "
            f"{rag_result['similarity']:.4f}"
        )


        print(

            "   Ticket:",

            rag_result[
                "instruction"
            ]
        )


        print(

            "   Category:",

            rag_result[
                "category"
            ]
        )


        print(

            "   Intent:",

            rag_result[
                "intent"
            ]
        )


    # ========================================================
    # BEST RETRIEVED RESULT
    # ========================================================

    best_result = (
        rag_results[0]
    )


    print(

        "\nBest similarity:",

        round(

            best_result[
                "similarity"
            ],

            4
        )
    )


    # ========================================================
    # DRAFT RESPONSE
    # ========================================================

    draft_response = (
        generate_draft_response(

            ticket,

            best_result
        )
    )


    print("\n" + "-" * 70)

    print(
        "RAG-BASED DRAFT RESPONSE"
    )

    print("-" * 70)


    print(
        draft_response
    )


    # ========================================================
    # HUMAN-IN-THE-LOOP
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "HUMAN-IN-THE-LOOP APPROVAL"
    )

    print("-" * 70)


    approval = input(

        "\nApprove this response? (y/n): "

    ).strip().lower()


    if approval == "y":

        print(
            "\nResponse APPROVED by human agent."
        )

        human_review = False


    else:

        print(

            "\nResponse NOT approved. "
            "Human agent should modify/review it."
        )

        human_review = True


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "ticket":
            ticket,

        "category":
            category,

        "category_confidence":
            category_confidence,

        "intent":
            intent,

        "intent_confidence":
            intent_confidence,

        "team":
            team,

        "human_review":
            human_review,

        "rag_results":
            rag_results,

        "draft_response":
            draft_response
    }


# ============================================================
# 19. DEMO TICKETS
# ============================================================

demo_tickets = [

    "I forgot my password",

    "My payment failed",

    "My card was charged twice",

    "Where is my order?"
]


# ============================================================
# 20. RUN DEMO
# ============================================================

print("\n")

print("=" * 70)

print(
    "RUNNING DEMO TICKETS"
)

print("=" * 70)


results = []


for ticket in demo_tickets:

    result = (
        process_ticket(
            ticket
        )
    )


    results.append(
        result
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n\n")

print("=" * 70)

print(
    "FINAL PIPELINE SUMMARY"
)

print("=" * 70)


for result in results:

    print("\nTicket:")

    print(
        result[
            "ticket"
        ]
    )


    print(

        "Category:",

        result[
            "category"
        ],

        f"({result['category_confidence']:.2%})"
    )


    print(

        "Intent:",

        result[
            "intent"
        ],

        f"({result['intent_confidence']:.2%})"
    )


    print(

        "Assigned Team:",

        result[
            "team"
        ]
    )


    if result[
        "human_review"
    ]:

        print(
            "Status: HUMAN REVIEW REQUIRED"
        )

    else:

        print(
            "Status: PROCESSED"
        )


print("\n")

print("=" * 70)

print(
    "PIPELINE COMPLETE"
)

print("=" * 70)