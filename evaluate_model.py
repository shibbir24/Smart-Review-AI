from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import numpy as np
import pandas as pd
import re
from collections import Counter

# ------------------ Review Generation ------------------
def generate_review(base_model, product, category, features, rating, tone, review_cache=None):
    """
    Generate a product review using LoRA fine-tuned model and apply repetition control.
    Optionally evaluates performance every 10 reviews.
    """
    adapter_path = "./lora_adapter"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompt = (
        f"Product: {product}\n"
        f"Category: {category}\n"
        f"Features: {features}\n"
        f"Rating: {rating}\n"
        f"Tone: {tone}\n\nReview:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.8,
            no_repeat_ngram_size=3,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------- Optional: Evaluation Trigger --------
    if review_cache is not None:
        review_cache.append(generated_text)
        if len(review_cache) % 10 == 0:
            metrics = compute_metrics(review_cache, requested_tone=tone)
            diversity = distinct_n_score(review_cache)
            metrics["distinct_n"] = diversity
            print(f"\n📊 Auto Evaluation after {len(review_cache)} reviews:")
            print(metrics)

    return generated_text


# ------------------ Evaluation Metrics ------------------
def compute_metrics(reviews, requested_tone="neutral"):
    """
    Compute simple text-level metrics:
    - avg_length: average word count
    - tone_match_ratio: how often requested tone appears
    """
    avg_length = np.mean([len(r.split()) for r in reviews]) if reviews else 0
    tone_match = sum(1 for r in reviews if re.search(requested_tone, r, re.IGNORECASE))
    tone_match_ratio = tone_match / len(reviews) if reviews else 0.0
    return {
        "avg_length": round(avg_length, 2),
        "tone_match_ratio": round(tone_match_ratio, 3)
    }


# ------------------ Diversity Metric ------------------
def distinct_n_score(texts, n=2):
    """
    Compute Distinct-N score (uniqueness measure).
    High values mean less repetition.
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        all_ngrams.extend(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    if not all_ngrams:
        return 0.0
    unique_ngrams = len(set(all_ngrams))
    return round(unique_ngrams / len(all_ngrams), 3)


# ------------------ Perplexity Evaluation ------------------
def evaluate_perplexity(base_model, test_csv="dataset/amazon_product_reviews.csv"):
    """
    Compute perplexity on a small subset of test data.
    Lower perplexity = better model.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, "./lora_adapter")
    model.eval()

    df = pd.read_csv(test_csv)
    texts = df["Review"].dropna().sample(min(50, len(df))).tolist()

    total_loss, total_tokens = 0, 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        total_loss += loss * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

    ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return round(ppl, 2)
