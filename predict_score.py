from pydantic import BaseModel, Field
from typing import Optional
import json
from datetime import datetime
import requests

# -------------------- Pydantic Models --------------------
class CustomerCreditLine(BaseModel):
    Id: str
    AccountId: str
    CompanyName: str
    ContactName: str
    Address: str
    Phone: str
    AccountOpenedSize: int  # Year opened
    CreditLimit: float
    CurrentBalance: float
    Country: str
    StateOrProvince: str
    City: str
    Zip: str

class CreditEvaluation(BaseModel):
    credit_score: float = Field(..., description="Predicted credit score (0-100)")
    eligible_loan_amount: float = Field(..., description="Suggested maximum loan amount based on credit score")

# -------------------- Gemini API Call --------------------
GEMINI_API_KEY = ""
GEMINI_ENDPOINT = "https://api.generativeai.google/v1beta2/models/gemini-1.5"  # Update if needed

def evaluate_credit(customer: CustomerCreditLine) -> CreditEvaluation:
    # Convert year to account age
    account_age = datetime.now().year - customer.AccountOpenedSize

    # Prepare prompt
    prompt = f"""
You are a financial AI model. Evaluate the following customer data and predict:
1. A credit score from 0 to 100.
2. Maximum eligible loan amount based on their current credit info.

Customer data (account age: {account_age} years):
{customer.json(indent=2)}

Return the response in JSON format exactly as:
{{
  "credit_score": float,
  "eligible_loan_amount": float
}}
"""

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "temperature": 0.2,
        "max_output_tokens": 200
    }

    # Send request to Gemini
    response = requests.post(
        f"{GEMINI_ENDPOINT}:predict",
        headers=headers,
        json=payload
    )
    
    response.raise_for_status()
    result_text = response.json()["candidates"][0]["content"]

    # Parse JSON
    result = json.loads(result_text)
    return CreditEvaluation(**result)

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    customer_data = {
        "Id":"2efd9d26-a571-4a1d-9e12-12362d7e8083",
        "AccountId":"de5d6cfb-5e3b-41c0-82c2-e6e670325dd3",
        "CompanyName":"Kennedy-Jackson",
        "ContactName":"Jeffery Morris",
        "Address":"50711 Mcdonald Street Suite 047, Reyeschester, AZ 04792",
        "Phone":"510-885-2362",
        "AccountOpenedSize":1994,
        "CreditLimit":96597.83,
        "CurrentBalance":29218.82,
        "Country":"Liechtenstein",
        "StateOrProvince":"Illinois",
        "City":"North Anthonyton",
        "Zip":"38392"
    }

    customer = CustomerCreditLine(**customer_data)
    evaluation = evaluate_credit(customer)
    print(evaluation.json(indent=2))
