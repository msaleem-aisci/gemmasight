# GemmaSight: Multimodal Retrieval-Augmented Pathology Assistant

## 1. Abstract
GemmaSight is a multimodal AI pipeline designed to accelerate the classification of Microsatellite Instability (MSI-High) versus Microsatellite Stability (MSS) in colorectal cancer. By analyzing standard Hematoxylin and Eosin (H&E) stained tissue patches, the system bypasses the standard multi-week turnaround time for genomic biomarker testing. The architecture leverages feature fusion from dual vision foundation models, spatial interpretability, and Retrieval-Augmented Generation (RAG) to produce evidence-backed clinical pathology reports in real-time.

## 2. Dataset and Preprocessing
The model is trained and evaluated on a dataset of colorectal cancer H&E patches. 
* **Input Resolution:** Images are processed at $224 \times 224$ resolution, representing localized tissue crops at diagnostic magnification.
* **Feature Extraction Pipeline:** To optimize training compute, images are pre-processed through the foundation models, and the resulting high-dimensional embeddings are saved locally for downstream classifier training.

## 3. Model Architecture
The GemmaSight pipeline is built upon a composite architecture:

* **Dual-Encoder Feature Fusion:** The system processes the input patch through two distinct vision models:
    * *Google Path Foundation Model:* Captures domain-specific histological structures.
    * *MedSigLIP (bfloat16):* Captures fine-grained, cross-modal biomedical features.
    The outputs are concatenated to form a rich, unified mathematical representation of the tissue.
* **Classification Head:** A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout layers processes the fused embeddings to output a binary probability (MSI-High vs. MSS).
* **Spatial Interpretability (Occlusion Engine):** A sliding-window occlusion algorithm ($40 \times 40$ patch, stride 20) measures the drop in prediction confidence when specific regions are masked, generating a spatial heatmap of diagnostic relevance.

## 4. Case-Based Reasoning & Generation
To foster clinical trust, the system avoids providing isolated predictions by employing a RAG framework:

* **FAISS Vector Database:** Historical training embeddings are indexed using `faiss.IndexFlatL2`. At inference, the system calculates the Cosine Similarity to retrieve the top 3 most morphologically similar historical cases.
* **Generative Reporting (MedGemma 4B):** The original H&E patch is blended with the occlusion heatmap. This composite image, along with the quantitative prediction and the FAISS retrieval context, is passed to the MedGemma Vision-Language Model. MedGemma synthesizes this data to draft a professional, grounded pathology report explaining the visual evidence.

## 5. Validation and Results
The core classification network is evaluated on a held-out test set.
* **Evaluation Metrics:** Accuracy, Confusion Matrix, and Area Under the Receiver Operating Characteristic Curve (AUROC).
* **Clinical Interface:** The complete inference pipeline, from image upload to report generation, is deployed via a Gradio dashboard, allowing end-to-end processing in under two minutes per patient.
