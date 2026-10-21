"""Which regulations apply to which product. All live in Title 12 of the eCFR."""

REGULATIONS = {
    "1005": "Regulation E (Electronic Fund Transfers)",
    "1006": "Regulation F (Debt Collection Practices)",
    "1022": "Regulation V (Fair Credit Reporting)",
    "1024": "Regulation X (Real Estate Settlement Procedures)",
    "1026": "Regulation Z (Truth in Lending)",
    "1030": "Regulation DD (Truth in Savings)",
}

# Search is restricted to these parts for each predicted product.
# Federal student loans are governed by Title 34 (Department of Education), which
# is out of scope; Regulation Z covers private education loans only.
PRODUCT_PARTS = {
    "bank_account": ["1005", "1030"],
    "money_transfer": ["1005"],
    "credit_card": ["1026", "1005"],  # prepaid cards fall under Regulation E
    "credit_reporting": ["1022"],
    "debt_collection": ["1006"],
    "mortgage": ["1024", "1026"],
    "student_loan": ["1026"],
}
