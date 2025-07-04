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

Ensure you have the necessary libraries installed. We are using python 3.9

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 numpy==1.23.2 sentence-transformers==2.2.2 pytorch-tabnet==4.1.0 einops==0.6.0 timm==0.9.12 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Running a Prediction (Demonstration)

This script shows how to initialize the model, process raw patient data (text and numerical), and get a prediction.

**Note:** This example uses the **untrained, randomly initialized model** to demonstrate the data pipeline. The resulting predictions will be random and are for demonstration purposes only.

```python
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pytorch_tabnet.pretraining import TabNetPretrainer
from mma_et import MMA_ET

# --- 1. Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("CUDA is required for this script.")
NUM_TABULAR_FEATURES = 32
TEXT_EMBEDDING_DIM = 768  # bert-base-uncased embedding dimension
BATCH_SIZE = 1024
MAX_EPOCHS = 5

# Placeholder data for X_nb_train and X_nb_test
# Replace with your actual datasets
X_nb_train = np.random.rand(1000, NUM_TABULAR_FEATURES).astype(np.float32)  # 1000 samples
X_nb_test = np.random.rand(200, NUM_TABULAR_FEATURES).astype(np.float32)   # 200 samples

# Verify data
print("X_nb_train shape:", X_nb_train.shape, "type:", type(X_nb_train))
print("X_nb_test shape:", X_nb_test.shape, "type:", type(X_nb_test))
if X_nb_test.shape[0] == 0:
    raise ValueError("X_nb_test is empty. Provide a non-empty evaluation set.")

# --- 2. Initialize and Fit TabNetPretrainer ---
unsupervised_model = TabNetPretrainer(
    n_d=64,
    n_a=64,
    mask_type='sparsemax',
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=5e-5),
    device_name=device,  # Set device explicitly
    verbose=0
)

print("Fitting TabNetPretrainer...")
unsupervised_model.fit(
    X_train=X_nb_train,
    eval_set=[X_nb_test] if X_nb_test.shape[0] > 0 else None,
    pretraining_ratio=0.8,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=MAX_EPOCHS,
    num_workers=0,
    drop_last=False
)
print("TabNetPretrainer fitting complete.")

# --- 3. Initialize MMA_ET ---
model = MMA_ET(
    inp_dim=NUM_TABULAR_FEATURES,
    unsupervised_model=unsupervised_model
).to(device)

# Ensure group_attention_matrix is on cuda:0
model.enc_tabnet.group_attention_matrix = model.enc_tabnet.group_attention_matrix.to(device)

# Set the model to evaluation mode for inference
model.eval()
print(f"Model initialized on '{device}'.")

# --- 4. Prepare Input Data for a Single Patient ---
chief_complaint_text = "patient complains of severe chest pain and shortness of breath"

print("\nEncoding chief complaint text with BERT...")
text_encoder = SentenceTransformer("bert-base-uncased", device=device)
text_embedding = text_encoder.encode(chief_complaint_text, convert_to_numpy=True)

tabular_data = np.random.rand(1, NUM_TABULAR_FEATURES).astype(np.float32)
textual_tensor = torch.from_numpy(text_embedding).unsqueeze(0).to(device)
tabular_tensor = torch.from_numpy(tabular_data).to(device)

# --- 5. Run the Prediction Flow ---
print("\nRunning inference...")
with torch.no_grad():
    logits, _ = model(1, tabular_tensor, textual_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# --- 6. Display the Result ---
triage_zones = {0: "Green Zone", 1: "Yellow Zone", 2: "Red Zone"}
print(f"\n--- Prediction Results ---")
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