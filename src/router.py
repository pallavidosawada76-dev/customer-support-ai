# ============================================
# CUSTOMER SUPPORT AI - ROUTING
# ============================================


# Map each category to the appropriate support team

ROUTING_MAP = {

    "ACCOUNT": "Account Support",

    "PAYMENT": "Payments Team",

    "SHIPPING": "Shipping Team",

    "DELIVERY": "Delivery Team",

    "REFUND": "Refunds Team",

    "ORDER": "Order Support",

    "INVOICE": "Billing Team",

    "SUBSCRIPTION": "Subscription Team",

    "CANCEL": "Cancellation Team",

    "CONTACT": "Customer Support",

    "FEEDBACK": "Customer Feedback Team"
}


def route_ticket(category):
    """
    Takes a predicted category
    and returns the appropriate support team.
    """

    return ROUTING_MAP.get(
        category,
        "General Support"
    )