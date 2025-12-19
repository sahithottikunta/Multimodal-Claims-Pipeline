# Synthetic Tri-Modal Insurance Claims Predictor 

##  Project Overview
This repository contains the source code for a **Generative AI Pipeline** and **Real-Time Inference System** designed to automate car insurance claim cost estimation.

The project solves the "cold start" problem in insurance AI by synthetically generating a correlated dataset of 1,000 samples across three modalities:
1.  **Tabular Data:** Vehicle & Policy metadata (generated via CTGAN).
2.  **Textual Data:** Complaint narratives (generated via Llama 3).
3.  **Visual Data:** Damage imagery (generated via Stable Diffusion + LoRA).

##  Repository Contents
- **`Tri_Modal_Inference.ipynb`**: The main Jupyter Notebook containing:
    - Data Generation Logic (CTGAN, Llama 3, LoRA).
    - Model Training (Custom Tri-Modal ViT Architecture).
    - Inference API (Gradio Deployment).
- **`data/`**: The synthetic dataset used for training.
- **`requirements.txt`**: List of dependencies required to run the notebook.

##  Model Architecture
We implemented a custom **Tri-Modal Fusion Network** that integrates:
- **Vision:** Vision Transformer (ViT-Base) for image feature extraction.
- **Text:** TF-IDF Vectorization with Dense Layers for narrative analysis.
- **Tabular:** Fully Connected Layers for categorical metadata processing.

**Performance:** The model achieved a Validation $R^2$ Score of **0.880**, significantly outperforming baseline tabular models ($R^2=0.28$).

##  How to Run
1.  Clone this repository.
2.  Open `Tri_Modal_Inference.ipynb` in Google Colab (recommended for GPU support).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the notebook cells to train the model and launch the Gradio Interface.

##  Tech Stack
- **Frameworks:** PyTorch, Hugging Face Transformers, Gradio.
- **Models:** ViT-Base-Patch16-224, Llama 3, CTGAN.
- **Hardware:** Developed on NVIDIA A100 GPU (Google Colab).

---
**Author:** Ottikunta Sahith (Group 2)
