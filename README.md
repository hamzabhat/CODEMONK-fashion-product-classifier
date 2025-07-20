#  Fashion Product Classifier

A full-stack Machine Learning web application that predicts multiple attributes of fashion products from images using deep learning. This project was developed as part of a coding assignment for Codemonk’s Machine Learning Internship.

---

## 🚀 Project Overview

Given an input image of a fashion product, this system predicts the following four attributes:

-  **Color**
-  **Product Type**
-  **Season of Use**
-  **Target Gender**

It uses multi-task deep learning to handle all four tasks in a single model and provides visual heatmaps for interpretability.

---
## 🚧 Pending Improvements

While the core pipeline is complete—from preprocessing to prediction and visualization—several improvements are planned to enhance functionality, performance, and user experience:

- **🧠 Additional Architectures:**
    - [ ] Integrate Vision Transformer (ViT) with 4-head classification.
  - [ ] Experiment with Mixture-of-Experts (MoE) heads for advanced ensembling.

- **📊 Evaluation & Metrics:**
  - [ ] Add confusion matrices and classification reports per task.
- **🔧 Backend Enhancements:**
  - [ ] Add batch image support for predictions.
- **💡 Explainability:**
  - [ ] Improve Grad-CAM precision by using hooks on intermediate layers.
---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [How to Run This Project](#️how-to-run-this-project)
- [Example Output](#example-output)
- [Pending Improvements](#pending-improvements)
- [Model Checkpoints](#model-checkpoints)
- [Author](#author)


## 🧱 Project Architecture

### Multi-Head Learning Model 

- **Backbone**: `EfficientNetV2-S` (pretrained on ImageNet)
- **Shared Encoder**: Extracts rich image features
- **Four Classification Heads**:
  - Color
  - Product Type
  - Season
  - Gender
- **Regularization**: Dropout applied to each head
- **Loss**: Independent `CrossEntropyLoss` for each task
- **Training Framework**: PyTorch

> 🏆 This model design is efficient, generalizes well, and leverages shared representations to boost performance across tasks.

---
## Frontend

- **Framework**: [Streamlit](https://streamlit.io)
- **Features**:
  - Upload fashion product image
  - Displays predicted color, type, season, and gender
  - Shows visual heatmap of model attention

---

## Backend

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **APIs**:
  - `POST /predict`: Takes an image, returns predictions + heatmap path
- **Model Inference**:
  - Loads trained EfficientNet multi-task model
  - Decodes predicted labels using saved `label_encoders.json`

---
## Explainability

- **Grad-CAM** heatmaps are generated for each input image using the shared EfficientNet backbone.
- The heatmaps highlight which regions of the image were most influential in the predictions.

---



## Project Structure
```sh
fashion-product-classifier/
│
├── backend/
│ ├── main.py # FastAPI app entrypoint
│ ├── routes/ # API endpoints
│ ├── services/ # Prediction logic, heatmap gen
│ ├── utils/ # Preprocessing helpers
│ ├── models/ # Model architecture + checkpoints + encodings 
│ │ └── effi_net_backbone/
│ │ ├── model_efficientnet.pth
│ │ └── label_encoders.json
│ └── static/ # Heatmaps served here
│ └── heatmaps/
│
├── frontend/
│ └── app.py # Streamlit frontend UI
│
├── data/
│ ├── raw/
│ │ └── fashion-product-images-dataset/
│ └── processed/
│ ├── cleaned_styles.csv
│ └── distribution_plots/
│
├── data_preprocessing/
│ ├── clean_data.py # CSV cleaning
│ └── class_distribution.py # Plots label distributions
│
├── training/
│ ├── efficientnet_training.py # EfficientNet training script
│ └── vit_training.py # (NOT IMPLEMENTED YET) ViT training script
│
├── requirements.txt
└── README.md
```
## 🛠️ How to Run This Project

###  Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure you have:
> - Python 3.8+
> - PyTorch with GPU support (optional but recommended)

---

###  Step 2: Preprocess the Dataset

```bash
python data_preprocessing/clean_data.py
python data_preprocessing/class_distribution.py
```

---

###  Step 3: Train the Model
***Option 1*** — Train from scratch:
```bash
python training/efficientnet_training.py
```
This will save:
- Model weights to: `backend/models/effi_net_backbone/model_efficientnet.pth`
- Label encoders to: `backend/models/encodings/label_encoders.json`
Option 1 — Train from scratch:


***Option 2*** — Download pretrained:

👉 [Download pretrained model (.pth) + label encoders](#  Fashion Product Classifier

A full-stack Machine Learning web application that predicts multiple attributes of fashion products from images using deep learning. This project was developed as part of a coding assignment for Codemonk’s Machine Learning Internship.

---

## 🚀 Project Overview

Given an input image of a fashion product, this system predicts the following four attributes:

-  **Color**
-  **Product Type**
-  **Season of Use**
-  **Target Gender**

It uses multi-task deep learning to handle all four tasks in a single model and provides visual heatmaps for interpretability.

---
## 🚧 Pending Improvements

While the core pipeline is complete—from preprocessing to prediction and visualization—several improvements are planned to enhance functionality, performance, and user experience:

- **🧠 Additional Architectures:**
    - [ ] Integrate Vision Transformer (ViT) with 4-head classification.
  - [ ] Experiment with Mixture-of-Experts (MoE) heads for advanced ensembling.

- **📊 Evaluation & Metrics:**
  - [ ] Add confusion matrices and classification reports per task.
- **🔧 Backend Enhancements:**
  - [ ] Add batch image support for predictions.
- **💡 Explainability:**
  - [ ] Improve Grad-CAM precision by using hooks on intermediate layers.
---

## 📚 Table of Contents

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [🎥 Demo](#-demo)
- [🧪 Pending Improvements](#-pending-improvements)
- [🧠 Model Checkpoints](#-model-checkpoints)
- [👤 Author](#-author)



## 🧱 Project Architecture

### Multi-Head Learning Model 

- **Backbone**: `EfficientNetV2-S` (pretrained on ImageNet)
- **Shared Encoder**: Extracts rich image features
- **Four Classification Heads**:
  - Color
  - Product Type
  - Season
  - Gender
- **Regularization**: Dropout applied to each head
- **Loss**: Independent `CrossEntropyLoss` for each task
- **Training Framework**: PyTorch

> 🏆 This model design is efficient, generalizes well, and leverages shared representations to boost performance across tasks.

---
## Frontend

- **Framework**: [Streamlit](https://streamlit.io)
- **Features**:
  - Upload fashion product image
  - Displays predicted color, type, season, and gender
  - Shows visual heatmap of model attention

---

## Backend

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **APIs**:
  - `POST /predict`: Takes an image, returns predictions + heatmap path
- **Model Inference**:
  - Loads trained EfficientNet multi-task model
  - Decodes predicted labels using saved `label_encoders.json`

---
## Explainability

- **Grad-CAM** heatmaps are generated for each input image using the shared EfficientNet backbone.
- The heatmaps highlight which regions of the image were most influential in the predictions.

---



## Project Structure
```sh
fashion-product-classifier/
│
├── backend/
│ ├── main.py # FastAPI app entrypoint
│ ├── routes/ # API endpoints
│ ├── services/ # Prediction logic, heatmap gen
│ ├── utils/ # Preprocessing helpers
│ ├── models/ # Model architecture + checkpoints + encodings 
│ │ └── effi_net_backbone/
│ │ ├── model_efficientnet.pth
│ │ └── label_encoders.json
│ └── static/ # Heatmaps served here
│ └── heatmaps/
│
├── frontend/
│ └── app.py # Streamlit frontend UI
│
├── data/
│ ├── raw/
│ │ └── fashion-product-images-dataset/
│ └── processed/
│ ├── cleaned_styles.csv
│ └── distribution_plots/
│
├── data_preprocessing/
│ ├── clean_data.py # CSV cleaning
│ └── class_distribution.py # Plots label distributions
│
├── training/
│ ├── efficientnet_training.py # EfficientNet training script
│ └── vit_training.py # (NOT IMPLEMENTED YET) ViT training script
│
├── requirements.txt
└── README.md
```
## 🛠️ How to Run This Project

###  Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure you have:
> - Python 3.8+
> - PyTorch with GPU support (optional but recommended)

---

###  Step 2: Preprocess the Dataset

```bash
python data_preprocessing/clean_data.py
python data_preprocessing/class_distribution.py
```

---

###  Step 3: Train the Model
***Option 1*** — Train from scratch:
```bash
python training/efficientnet_training.py
```
This will save:
- Model weights to: `backend/models/effi_net_backbone/model_efficientnet.pth`
- Label encoders to: `backend/models/encodings/label_encoders.json`
Option 1 — Train from scratch:


***Option 2*** — Download pretrained:

👉 [Download pretrained model (.pth) + label encoders](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

Then place them inside:

```
backend/models/
├── effi_net_backbone/model_efficientnet.pth
└── encodings/label_encoders.json
```
---

###  Step 4: Run Backend API (FastAPI)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

> The API will be live at: [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

###  Step 5: Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

> The UI will open in your browser → Upload an image → Get predictions + visual heatmap.

## Demo

[Demo](assets/demo.gif)

> Predictions are returned with corresponding Grad-CAM heatmap.
---
## 📦 Model Checkpoints

You can download the pre-trained model and label encoders here:

👉 [EfficientNetV2 Backbone Checkpoint (.pth) + Label Encoders](https://drive.google.com/drive/folders/1xQVRHiBioyCYDghrJk-5HKf7c6vZt_Z9?usp=drive_link)

> Place the downloaded files in:
> ```
> backend/models/
> ├── effi_net_backbone/model_efficientnet.pth
> └── encodings/label_encoders.json
> ```

---
## 👨‍💻 Author

**Mohammad Hamza Bhat**  
ML Intern Candidate @ Codemonk  
📧 Email: [hamzabhat88@gmail.com](mailto:hamzabhat88@gmail.com)   
Phone: +91 9797860227
)

Then place them inside:

```
backend/models/
├── effi_net_backbone/model_efficientnet.pth
└── encodings/label_encoders.json
```
---

###  Step 4: Run Backend API (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```

> The API will be live at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

###  Step 5: Run Frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

> The UI will open in your browser → Upload an image → Get predictions + visual heatmap.

## Example Output

|  Input Image |  Predictions |  Heatmap |
|----------------|----------------|------------|
| ![product](https://via.placeholder.com/150x150?text=Image) |  **Color**: Blue<br> **Type**: T-shirts<br> **Season**: Summer<br> **Gender**: Unisex | ![heatmap](https://via.placeholder.com/150x150?text=GradCAM) |

> Predictions are returned with corresponding Grad-CAM heatmap.
---
## 📦 Model Checkpoints

You can download the pre-trained model and label encoders here:

👉 [EfficientNetV2 Backbone Checkpoint (.pth) + Label Encoders](https://drive.google.com/drive/folders/1xQVRHiBioyCYDghrJk-5HKf7c6vZt_Z9?usp=drive_link)

> Place the downloaded files in:
> ```
> backend/models/
> ├── effi_net_backbone/model_efficientnet.pth
> └── encodings/label_encoders.json
> ```

---
## 👨‍💻 Author

**Mohammad Hamza Bhat**  
ML Intern Candidate @ Codemonk  
📧 Email: [hamzabhat88@gmail.com](mailto:hamzabhat88@gmail.com)   
Phone: +91 9797860227
