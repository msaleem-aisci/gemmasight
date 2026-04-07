# GemmaSight: Multimodal Retrieval-Augmented Pathology Assistant

## 1. Abstract
GemmaSight is an end-to-end multimodal AI pipeline engineered to accelerate the classification of Microsatellite Instability (MSI-High) versus Microsatellite Stability (MSS) in colorectal cancer. By analyzing standard Hematoxylin and Eosin (H&E) stained tissue patches, the system bypasses the standard multi-week turnaround time for genomic biomarker testing. The architecture leverages feature fusion from dual vision foundation models, spatial interpretability (Occlusion Heatmaps), and Retrieval-Augmented Generation (RAG) to produce evidence-backed clinical pathology reports in under two minutes.

## 2. Clinical Motivation
In colorectal cancer oncology, determining a patient's MSI status is a critical prerequisite for administering targeted immunotherapy. The current clinical gold standard relies on Polymerase Chain Reaction (PCR) or Next-Generation Sequencing (NGS). While accurate, these genomic tests introduce a two-to-four-week diagnostic delay. 

GemmaSight operates on the premise that distinct genomic mutations manifest as recognizable morphological phenotypes at the cellular level. By interrogating the readily available H&E slides from the initial biopsy, this system aims to instantly triage patients, allowing clinical teams to initiate preliminary treatment planning weeks ahead of genomic confirmation.

## 3. Dataset and Preprocessing
The model is trained and evaluated on a highly curated dataset of colorectal cancer H&E patches. 

* **Input Resolution:** Whole Slide Images (WSIs) are pre-processed into localized tissue crops at $224 \times 224$ resolution. This represents diagnostic-level magnification required to observe cellular and nuclear atypia.
* **Feature Extraction Pipeline:** To optimize computational resources during the hyperparameter tuning of the classifier, raw images are not passed through the foundation models during every epoch. Instead, embeddings are pre-extracted, fused, and saved as high-dimensional `.npy` arrays.
* **Class Imbalance:** The pipeline accounts for the natural biological prevalence of MSS over MSI-High through strategic batch generation and evaluation metrics focused on the Area Under the Receiver Operating Characteristic Curve (AUROC) rather than raw accuracy.

## 4. System Architecture
GemmaSight is built upon a highly modular, five-stage architecture.

### 4.1. Dual-Encoder Feature Fusion
The system captures orthogonal visual information by passing the input patch through two distinct, frozen vision foundation models:
1. **Google Path Foundation Model:** Optimized specifically for histological and morphological structures inherent to H&E stains.
2. **MedSigLIP (bfloat16):** A vision-language encoder that captures fine-grained, cross-modal biomedical features. 

The resulting vectors are concatenated along the feature dimension to form a unified, high-dimensional representation of the tissue sample.

### 4.2. Classification Engine
The fused embeddings are processed by a Multi-Layer Perceptron (MLP) designed for robust binary classification.
* **Architecture:** Three dense hidden layers (512 $\rightarrow$ 256 $\rightarrow$ 128 units) utilizing ReLU activations.
* **Regularization:** Heavy regularization is applied via Batch Normalization and Dropout (0.4, 0.3, 0.2) to prevent overfitting on the high-dimensional input space.
* **Output:** A final sigmoid activation yields the continuous probability score of the tissue exhibiting an MSI-High phenotype.

### 4.3. Case-Based Reasoning (FAISS)
To foster clinical trust, GemmaSight avoids isolated "black-box" predictions. It employs a $k$-Nearest Neighbors retrieval system to find historical morphological matches.
* **Vector Database:** Historical training embeddings are L2-normalized and indexed using `faiss.IndexFlatL2`. 
* **Retrieval Metric:** The system calculates the Cosine Similarity between the query vector $Q$ and database vectors $D$:
<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{Similarity}(Q,D)=\frac{Q\cdot&space;D}{\|Q\|\|D\|}" title="Cosine Similarity Formula" />
</div>
* **Execution:** At inference, the system retrieves the top 3 most morphologically similar historical cases to serve as clinical evidence.

### 4.4. Spatial Interpretability (Occlusion Engine)
To localize the specific cellular structures driving the MLP's prediction, a sliding-window occlusion algorithm is deployed.
* A $40 \times 40$ neutral patch with a stride of 20 is systematically passed across the image.
* The system measures the delta in prediction confidence when specific regions are masked.
* These deltas are aggregated, smoothed via Gaussian blurring, and thresholded to generate a spatial saliency heatmap highlighting the top 20% most diagnostically relevant regions.

### 4.5. Generative Synthesis (MedGemma 4B)
The final stage bridges quantitative AI with qualitative clinical reporting.
* The original H&E patch is visually blended with the occlusion heatmap.
* This composite image, the quantitative prediction risk, and the textual FAISS retrieval context are injected into a highly constrained prompt.
* **MedGemma 4B-IT** synthesizes this multimodal context to draft a professional, grounded pathology report explaining the visual evidence and correlating it with the historical database matches.

## 5. Deployment and Usage
The complete inference pipeline is deployed via a Gradio dashboard, allowing clinicians to upload an image and receive the full multi-modal analysis—including heatmaps, historical case retrievals, and the generated report—in a single interface.

### Requirements
```bash
pip install torch transformers accelerate huggingface_hub
pip install faiss-cpu opencv-python Gradio umap-learn
