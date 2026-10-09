from router import route_ticket


test_categories = [
    "ACCOUNT",
    "PAYMENT",
    "SHIPPING",
    "REFUND",
    "DELIVERY",
    "ORDER"
]


print("========== ROUTING TEST ==========")

for category in test_categories:

    team = route_ticket(category)

    print(
        "Category:",
        category,
        "→ Team:",
        team
    )