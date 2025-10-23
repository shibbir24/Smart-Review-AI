import streamlit as st
from finetune_lora import train_lora
from evaluate_model import generate_review, compute_metrics, evaluate_perplexity
import matplotlib.pyplot as plt
import time
import os

st.set_page_config(page_title="LLM LoRA Fine-Tuning App", layout="wide")
st.title("🧠 LoRA Fine-Tuned LLM for Product Review Generation & Evaluation")

tab1, tab2 = st.tabs(["⚙️ Fine-Tuning", "🪶 Generate & Evaluate Reviews"])

# ----------- TAB 1: FINE-TUNING -----------
with tab1:
    st.header("⚙️ Fine-Tune Model with LoRA")

    model_choice = st.selectbox("Select Base Model", ["gpt2", "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B"])
    epochs = st.slider("Epochs", 1, 5, 2)
    lr = st.number_input("Learning Rate", value=1e-4, step=1e-5, format="%.5f")

    metrics = None

    if st.button("🚀 Start Fine-Tuning"):
        st.info(f"Starting LoRA fine-tuning on `{model_choice}` for {epochs} epochs...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(epochs):
            status_text.text(f"Training... Epoch {epoch+1}/{epochs}")
            time.sleep(1.5)  # simulate training progress
            progress_bar.progress(int((epoch + 1) / epochs * 100))

        with st.spinner("Finalizing and saving adapter..."):
            metrics = train_lora(model_choice, epochs, lr)

        st.success("✅ Fine-tuning Completed Successfully!")
        st.balloons()

        if metrics:
            st.write("### 📊 Training Summary")
            st.json(metrics)

            if "train_loss" in metrics:
                fig, ax = plt.subplots()
                ax.plot(metrics["train_loss"], label="Training Loss", marker='o')
                ax.set_xlabel("Steps")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

        progress_bar.empty()
        status_text.text("Training finished ✔️")

# ----------- TAB 2: GENERATION & EVALUATION -----------
with tab2:
    st.header("🪶 Generate & Evaluate Product Reviews")

    model_choice = st.selectbox("Select Fine-Tuned Model", ["gpt2", "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B"], key="gen_model")

    product = st.text_input("Product Name", "Amazon Kindle E-Reader")
    category = st.text_input("Category", "E-Reader")
    features = st.text_area("Features", "6-inch display, WiFi, lightweight")
    rating = st.slider("Rating", 1, 5, 5)
    tone = st.selectbox("Tone", ["enthusiastic", "critical", "balanced"])

    if "generated_reviews" not in st.session_state:
        st.session_state.generated_reviews = []

    if st.button("✨ Generate Review"):
        with st.spinner("Generating review..."):
            review = generate_review(model_choice, product, category, features, rating, tone)
        st.success("✅ Generated Review:")
        st.write(review)

        st.session_state.generated_reviews.append(review)
        count = len(st.session_state.generated_reviews)
        st.info(f"🧩 Total Generated Reviews: {count}")

        # --------- Evaluation Dashboard (Tracking) ---------
        st.subheader("📊 Evaluation Dashboard")

        # Initialize storage for evaluation history
        if "evaluation_history" not in st.session_state:
            st.session_state.evaluation_history = []

        # -------- Evaluate every 10 reviews --------
        if count % 10 == 0:
            st.warning("📊 Evaluating model performance after 10 reviews...")
            with st.spinner("Running metrics evaluation..."):
                metrics = compute_metrics(st.session_state.generated_reviews, requested_tone=tone)
                ppl = evaluate_perplexity(model_choice, test_csv="dataset/amazon_product_reviews.csv")

            st.subheader("📈 Evaluation Metrics")
            st.write(f"- **Average Length:** {metrics['avg_length']:.2f} words")
            st.write(f"- **Tone Match Ratio:** {metrics['tone_match_ratio']*100:.1f}%")
            st.write(f"- **Perplexity (↓ better):** {ppl:.2f}")

            # Save current evaluation data
            eval_data = {
                "reviews_count": count,
                "avg_length": metrics["avg_length"],
                "tone_match": metrics["tone_match_ratio"] * 100,
                "perplexity": ppl
            }
            st.session_state.evaluation_history.append(eval_data)

        import pandas as pd
        df = pd.DataFrame(st.session_state.evaluation_history)

        # Show line trends over evaluations
        if not df.empty:
            st.line_chart(df.set_index("reviews_count")[["avg_length", "tone_match", "perplexity"]])
            st.dataframe(df.style.format({"avg_length": "{:.2f}", "tone_match": "{:.1f}", "perplexity": "{:.2f}"}))

        st.markdown("✅ Model evaluation updates automatically after every 10 generated reviews — showing live performance improvements.")


