# MedFusionNet: Hybrid Transformer-Based Multimodal Deep Learning for Women's Health

Official implementation of **MedFusionNet**, a multimodal framework designed for the early detection of Breast Cancer, Cervical Cancer, and Polycystic Ovary Syndrome (PCOS) by integrating medical imaging (Ultrasound/Mammography) with textual clinical reports.

This repository addresses the clinical need for model interpretability, statistical robustness (N=1,129), and reproducible multimodal fusion.

---

## 🚀 Features
- **Two-Stage Training**: Backbone-frozen initialization followed by end-to-end fine-tuning.
- **CMAF Module**: Cross-Modal Attention Fusion to align visual ROIs with clinical keywords.
- **Multi-Task Heads**: Specialized classification heads for BC, CC, and PCOS.
- **Model Checkpointing**: Automatic saving of the best-performing weights.

---

## 📂 Project Structure
```text
MedFusionNet/
├── config.yaml          # Hyperparameters and Dataset Paths
├── requirements.txt     # Dependency list
├── model.py             # MedFusionNet Architecture (CMAF)
├── dataset.py           # Multi-task Data Loading Logic
├── train.py             # Two-stage Training Script
└── README.md            # Step-by-step Execution Guide

🛠️ Installation  
Clone the repository:  
code  
Bash  
git clone https://github.com/your-username/MedFusionNet.git  
cd MedFusionNet  

Install dependencies:  
code  
Bash  
pip install -r requirements.txt  

📊 Dataset Configuration  
The model is pre-configured to utilize standard Kaggle dataset paths. Ensure the following datasets are available in your environment:  

Disease | Dataset Source | Default Kaggle Path  
Breast Cancer | CBIS-DDSM | /kaggle/input/cbis-ddsm-breast-cancer-image-dataset  
Cervical Cancer | 224-224 Screening | /kaggle/input/224-224-cervical-cancer-screening  
PCOS | Ultrasound Images | /kaggle/input/pcos-detection-using-ultrasound-images  

To change paths, edit the `paths` section in `config.yaml`.  

🧠 Model Architecture: CMAF  
The core of this framework is the Cross-Modal Attention Fusion (CMAF). It replaces simple concatenation with a Multi-Head Cross-Attention mechanism:  

- Query (Q): Extracted from the Vision Transformer (ViT) visual tokens.  
- Key (K) & Value (V): Extracted from the DistilBERT clinical text embeddings.  
- Outcome: The model "attends" to specific image regions based on the clinical keywords provided in the text, ensuring a context-aware diagnosis.  

🏋️ Training Guide  
Run the main training script to execute the two-stage optimization process:  

code  
Bash  
python train.py  

Stage 1: Backbone Frozen (70 Epochs)  
- Image/Text Encoders: Frozen to preserve pre-trained feature extraction.  
- Focus: Optimizing the CMAF fusion layer and the diagnostic classification heads.  
- Learning Rate: 1e-4 (AdamW).  

Stage 2: Full Fine-tuning (30 Epochs)  
- Image/Text Encoders: Unfrozen.  
- Focus: Global feature alignment across all modalities.  
- Learning Rate: 1e-5.  

💾 Model Artifacts  
The best-performing model weights (based on validation loss) are automatically saved to:  

./medfusionnet_best_model.pth  

Load the model using:  

code  
Python  
model.load_state_dict(torch.load('./medfusionnet_best_model.pth'))  

📜 Requirements  
- Python 3.8+  
- PyTorch 2.0+  
- Transformers (HuggingFace)  
- PyYAML  
- Pillow  
- OpenCV  
