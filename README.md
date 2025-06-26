# Multimodal Attention-based Deep Learning for Emergency Triage

This repository contains the model and code for the paper: **"Multimodal Attention-based Deep Learning for Emergency Triage with Electronic Health Records"**. Our model, `MMA_ET`, uses a Vision Transformer to process both tabular and textual patient data for accurate triage prediction.

## Overview

This project provides a deep learning model for emergency triage. The `MMA_ET` architecture integrates a `TabNetEncoder` for structured vital signs and text embeddings for unstructured chief complaints, feeding the combined features into a `Vision Transformer (ViT)` for robust prediction.

This README provides a complete example of how to use the model's architecture, showing the end-to-end data flow from raw inputs to a final prediction.

## Requirements

Ensure you have the necessary libraries installed.

```bash
pip install torch numpy pytorch-tabnet timm einops sentence-transformers
```

## Running a Prediction (Demonstration)

This script shows how to initialize the model, process raw patient data (text and numerical), and get a prediction.

**Note:** This example uses the **untrained, randomly initialized model** to demonstrate the data pipeline. The resulting predictions will be random and are for demonstration purposes only.

```python
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# For model architecture initialization
from pytorch_tabnet.pretraining import TabNetPretrainer

# Import your custom model class from your project files
# (Make sure mma_et.py and vit_orig.py are in the same directory)
from mma_et import MMA_ET

# --- 1. Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TABULAR_FEATURES = 32
TEXT_EMBEDDING_DIM = 768 # bert-base-uncased embedding dimension

# --- 2. Initialize Model Architecture ---

# The MMA_ET model requires a TabNetPretrainer object for its structure.
# We just need to instantiate it; no fitting is required for this demo.
unsupervised_model = TabNetPretrainer(
    n_d=64, n_a=64, # Params should match your model's training config
    mask_type='sparsemax'
)

# Initialize the model architecture
model = MMA_ET(
    inp_dim=NUM_TABULAR_FEATURES,
    unsupervised_model=unsupervised_model
).to(device)

# In a real-world scenario, you would load your trained weights here.
# For example:
#   MODEL_WEIGHTS_PATH = "path/to/your/weights.pth"
#   model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

# Set the model to evaluation mode
model.eval()
print(f"Model initialized on '{device}'. Running with random weights.")


# --- 3. Prepare Input Data for a Single Patient ---

# a) Raw text data (patient's chief complaint)
chief_complaint_text = "fever for two days associated with shortness of breath for one week"

# b) Load the SentenceTransformer model to create the text embedding
print("\nEncoding chief complaint text...")
text_encoder = SentenceTransformer("bert-base-uncased", device=device)
text_embedding = text_encoder.encode(chief_complaint_text, convert_to_numpy=True)

# c) Preprocessed tabular data (vital signs, etc.)
# We use random data here as a placeholder for a real patient's data.
tabular_data = np.random.rand(1, NUM_TABULAR_FEATURES).astype(np.float32)

# d) Convert numpy arrays to PyTorch tensors and move to the correct device
textual_tensor = torch.from_numpy(text_embedding).unsqueeze(0).to(device)
tabular_tensor = torch.from_numpy(tabular_data).to(device)


# --- 4. Run the Prediction Flow ---

print("\nRunning inference...")
with torch.no_grad(): # Disable gradient calculation
    # The model expects a batch_size argument (1 for a single prediction)
    logits, _ = model(1, tabular_tensor, textual_tensor)

    # Apply softmax to convert raw logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Get the predicted class index by finding the max probability
    predicted_class = torch.argmax(probabilities, dim=1).item()


# --- 5. Display the (Random) Result ---

# Triage zones are usually 0: Green (Non-urgent), 1: Yellow (Urgent), 2: Red (Critical)
triage_zones = {0: "Green Zone", 1: "Yellow Zone", 2: "Red Zone"}

print(f"\n--- Prediction Results (from Untrained Model) ---")
print(f"Input Text: '{chief_complaint_text}'")
print(f"Probabilities (Green, Yellow, Red): {probabilities.cpu().numpy().flatten()}")
print(f"Predicted Class Index: {predicted_class}")
print(f"--> Predicted Triage Level: '{triage_zones.get(predicted_class, 'Unknown')}'")

```