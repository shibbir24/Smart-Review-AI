"""
Streamlit App: Product Review Generator using Groq LLM
Features:
- Select from pre-defined examples or enter custom inputs
- Sidebar for model settings (model_id, temperature, top-p, max tokens)
- Calls Groq API and extracts structured generated review
"""

import streamlit as st
import os
import requests
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/responses"  # Groq OpenAI-compatible endpoint

# ---------------------------
st.set_page_config(page_title="Product Review Generator", layout="centered")
st.title("📝 AI Product Review Generator")
st.markdown(
    """
This demo generates synthetic product reviews using advanced **Groq LLMs**.
**Important:** Do not use generated reviews to impersonate real customers. Label clearly as *synthetic/demo*.
"""
)

# ---------------------------
# Sidebar: Model settings
with st.sidebar:
    st.header("Settings")

    # Model selection
    model_options = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b"
    ]
    model_id = st.selectbox("Select Groq model", model_options, index=model_options.index(DEFAULT_MODEL_ID))

    # Generation parameters
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8)
    top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 0.92)
    max_new_tokens = st.slider("Max tokens (generation)", 10, 400, 150)

# ---------------------------
# Pre-defined example prompts
example_prompts = [
    {
        "label": "Headphones",
        "product": "Wireless Headphones X200",
        "category": "Electronics",
        "features": "noise-cancelling, 30hr battery",
        "reviewer": "25-34, frequent traveller",
        "tone": "balanced"
    },
    {
        "label": "Blender",
        "product": "Kitchen Blender Pro",
        "category": "Home & Kitchen",
        "features": "1000W motor, compact, dishwasher-safe",
        "reviewer": "35-44, home cook",
        "tone": "enthusiastic"
    },
    {
        "label": "Smartwatch",
        "product": "FitWatch 5",
        "category": "Wearables",
        "features": "heart-rate monitor, waterproof, sleep tracking",
        "reviewer": "18-24, fitness enthusiast",
        "tone": "critical"
    },
]

# Dropdown to select example
selected_label = st.selectbox(
    "Select an example prompt",
    [ex["label"] for ex in example_prompts],
    index=0
)

# Populate input fields from selected example
selected_example = next(ex for ex in example_prompts if ex["label"] == selected_label)
product = st.text_input("Product title", value=selected_example["product"]).strip()
category = st.text_input("Category", value=selected_example["category"]).strip()
features = st.text_area("Features (comma separated)", value=selected_example["features"]).strip()
tone = st.selectbox(
    "Tone",
    ["enthusiastic", "balanced", "critical"],
    index=["enthusiastic", "balanced", "critical"].index(selected_example["tone"])
)
reviewer = st.text_input("Reviewer profile", value=selected_example["reviewer"]).strip()

# ---------------------------
# Function: Extract generated text from Groq response
def extract_groq_text(resp_json):
    try:
        for item in resp_json.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "").strip()
        return "[No text found]"
    except Exception as e:
        return f"[Error parsing response: {e}]"

# ---------------------------
# Generate review
if st.button("Generate Review"):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set.")
    else:
        prompt = (
            f"Product: {product}\nCategory: {category}\nFeatures: {features}\n"
            f"Reviewer: {reviewer}\nTone: {tone}\n\nReview:\n"
        )
        st.markdown("**Prompt:**")
        st.code(prompt)

        with st.spinner("Generating..."):
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model_id,
                    "input": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_new_tokens
                }
                response = requests.post(GROQ_ENDPOINT, headers=headers, json=data)
                response.raise_for_status()
                resp_json = response.json()

                generated_text = extract_groq_text(resp_json)
                st.markdown("---")
                st.subheader("📝 Generated Review")
                st.markdown(f"> {generated_text}")
                st.markdown("---")

            except Exception as e:
                st.error(f"Groq API request failed: {e}")
