import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import streamlit as st

def train_lora(base_model: str, epochs: int = 2, lr: float = 1e-4, train_csv: str = "dataset/amazon_product_reviews.csv"):
    """
    Fine-tune a base model using LoRA on the provided dataset and visualize progress in Streamlit.
    """
    st.write(f"### 🔧 Loading base model `{base_model}`...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    st.info("📂 Loading dataset for fine-tuning...")
    ds = load_dataset("csv", data_files={"train": train_csv})["train"]

    def preprocess(example):
        prompt = (
            f"Product: {example.get('Product','')}\n"
            f"Category: {example.get('Category','')}\n"
            f"Features: {example.get('Features','')}\n"
            f"Rating: {example.get('Rating','')}\n"
            f"Tone: {example.get('Tone','')}\n\n"
            f"Review: {example.get('Review','')}"
        )
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=256)

    tokenized_ds = ds.map(preprocess, batched=False)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to base model
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = get_peft_model(model, lora_config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = "./lora_adapter"
    os.makedirs(output_dir, exist_ok=True)

    # Streamlit progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    loss_list = []

    from transformers import TrainerCallback
    class StreamlitCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                loss_list.append(loss)
                progress = int((state.epoch / epochs) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Epoch {state.epoch:.1f}/{epochs} | Step {state.global_step} | Loss: {loss:.4f}")
                loss_chart.line_chart(loss_list)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[StreamlitCallback()]
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    st.success("🎉 LoRA adapter trained and saved successfully!")
    return {"train_loss": loss_list, "epochs": epochs, "base_model": base_model}
