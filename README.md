# Multimodal Attention-based Deep Learning for Emergency Triage

This repository contains the model and code for the paper: **"Multimodal Attention-based Deep Learning for Emergency Triage with Electronic Health Records"**. Our model, `MMA_ET`, uses a Vision Transformer to process both tabular and textual patient data for accurate triage prediction.

## Overview

This project provides a deep learning architecture designed for accurate and automated emergency triage. The core of the `MMA_ET` model is its ability to process multiple data types simultaneously to form a comprehensive understanding of a patient's condition.

The model's workflow is as follows:
1.  **Textual Data:** The patient's unstructured chief complaint is converted into a meaningful numerical representation using a pre-trained **BERT** model, accessed via the `SentenceTransformer` library. This captures the semantic context of the patient's symptoms.
2.  **Tabular Data:** Structured data, such as vital signs and other numerical/categorical patient information, is processed using a `TabNetEncoder`.
3.  **Feature Fusion & Prediction:** The embeddings from both BERT and TabNet are combined and fed into a `Vision Transformer (ViT)`, which excels at identifying complex patterns and global relationships between all features to make a final triage level prediction.

This README provides a complete, runnable example demonstrating this entire data flow, from raw inputs to a final (random) prediction using the untrained model architecture.

## Requirements

Ensure you have the necessary libraries installed.

```bash
pip install torch numpy
pip install pytorch-tabnet timm einops
pip install sentence-transformers
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

# Initialize the main model architecture
model = MMA_ET(
    inp_dim=NUM_TABULAR_FEATURES,
    unsupervised_model=unsupervised_model
).to(device)

# In a real-world scenario, you would load your trained weights here.
# For example:
#   MODEL_WEIGHTS_PATH = "path/to/your/weights.pth"
#   model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

# Set the model to evaluation mode for inference
model.eval()
print(f"Model initialized on '{device}'. Running with random weights.")


# --- 3. Prepare Input Data for a Single Patient ---

# a) Raw text data (patient's chief complaint)
chief_complaint_text = "patient complains of severe chest pain and shortness of breath"

# b) Use SentenceTransformer (BERT) to create the text embedding
# In our pipeline we do some preprocessing like mentioned in the paper
print("\nEncoding chief complaint text with BERT...")
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
with torch.no_grad(): # Disable gradient calculation for efficiency
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

## Acknowledgments

This work was supported by Universiti Sains Malaysia (Grant number 1001/PPSP/8014125).

## Authors

* **Hazqeel Afyq Athaillah Kamarul Aryffin** - *School of Computer Sciences, Universiti Sains Malaysia*
* **Kamarul Aryffin Baharuddin** - *School of Medical Sciences, Universiti Sains Malaysia*
* **Mohd Halim Mohd Noor** - *School of Computer Sciences, Universiti Sains Malaysia*