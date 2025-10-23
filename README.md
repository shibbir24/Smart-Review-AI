# Smart-Review-AI: Fine-Tuning GPT-2 LLM with LoRA

Generate realistic, human-like product reviews using a fine-tuned GPT-2 LLM model enhanced with LoRA.

## 🚀 Project Overview

Smart Review AI is a lightweight, AI-powered application designed to generate human-like product reviews. It leverages a GPT-2 language model fine-tuned with LoRA (Low-Rank Adaptation) to produce coherent, context-aware, and natural-sounding reviews for any product.

This app demonstrates how LoRA can efficiently adapt large language models without the need to retrain the entire model, making it faster, cheaper, and easier to deploy.

## ✨ Features

- Humanized Review Generation – Generates realistic reviews with controllable tone and ratings.
- Fine-Tuning Flexibility – Easily fine-tune GPT-2 for new product categories or domains.
- Efficient & Lightweight – Uses LoRA adapters, keeping the model small and fast.
- Streamlit Interface – Intuitive web app for generating reviews and evaluating outputs.
- Hugging Face Spaces Deployment – Live demo available in the cloud.

## 🛠️ Tech Stack

- Language Model: GPT-2
- Fine-Tuning: LoRA (Low-Rank Adaptation) via PEFT
- Web App: Streamlit
- Deployment: Hugging Face Spaces
- Python Libraries: transformers, peft, torch, numpy, pandas

## 💻 How It Works

- Select a Base Model – GPT-2 is loaded as the base LLM.
- Apply LoRA Adapter – Fine-tuned LoRA weights are loaded to adapt the model to product reviews.
- Start Fine-tuning with your dataset
- Generate Review – Enter product details, category, features, rating, and tone. The model generates a human-like review.
- Reviews Evaluation – Track average words length, Tune match ratio, and perplexity.

🖥️ Live Demo

Check out the web app on Hugging Face Spaces:

https://huggingface.co/spaces/shibbir24/SmartReviewAI

## ⚡Usage Example
```
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

### Load base model and LoRA adapter
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(model, "lora_adapter")
model.eval()

### Generate review
inputs = tokenizer("Product: Amazon Kindle\nCategory: E-Reader\nFeatures: Lightweight, long battery life\nRating: 5 stars\nTone: Enthusiastic", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))
```

## 📂 Repository Structure
```
Smart-Review-AI/
│
├── app.py               # Streamlit web app
├── lora_adapter/        # LoRA adapter files (.safetensors + config)
├── requirements.txt     # Required Python packages
├── evaluate_model.py    # Model reviews evaluation
├── finetune_lora.py     # Model fine-tuning
└── README.md            # Project documentation
```

## 💡 Why LoRA?

- Fine-tuning large language models traditionally requires gigabytes of storage and GPU resources.
- LoRA allows parameter-efficient fine-tuning, updating only a small subset of weights.
- Faster training, smaller model files, and seamless integration with GPT-2.

## 📝 License

This project is licensed under the MIT License – feel free to use, modify, and deploy it.

## 🙌 Contribution

Contributions, ideas, or improvements are welcome. Open an issue or submit a pull request to collaborate.

