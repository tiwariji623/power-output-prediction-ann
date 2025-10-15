# ⚡ Power Output Prediction using Artificial Neural Networks (ANN)

## 📌 Overview
This project demonstrates a deep learning approach to predicting the electrical power output of a Combined Cycle Power Plant (CCPP) using an Artificial Neural Network (ANN). The model is trained on real-world data and achieves high predictive accuracy, making it suitable for energy forecasting and operational optimization.

---

## 📊 Dataset
- **Source**: [UCI Machine Learning Repository – CCPP Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- **Samples**: 9,568 observations
- **Features**:
  - Ambient Temperature (AT)
  - Ambient Pressure (AP)
  - Relative Humidity (RH)
  - Exhaust Vacuum (V)
- **Target**: Electrical energy output (PE)

---

## 🧠 Model Architecture
- Framework: **Keras with TensorFlow backend**
- Architecture:
  - Input layer with 4 neurons
  - Two hidden layers with ReLU activation
  - Output layer with linear activation
- Optimizer: **Adam**
- Loss function: **Mean Squared Error (MSE)**

---

## 📈 Performance Highlights
- **R² Score**: `0.93` — excellent predictive power
- **Training MSE**: `0.0032`  
- **Validation MSE**: `0.0037`

> The model explains 93% of the variance in power output, demonstrating strong generalization across unseen data.

### 🔍 Visual Insights
- 📉 Training vs. Validation loss curves  
- 📊 Predicted vs. Actual scatter plot  
- 📐 Residual distribution

---

## 🗂️ Repository Structure

| File/Folder                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `power_output_prediction_ann.ipynb` | Main notebook with full workflow: data loading, preprocessing, model training, evaluation, and visualization |
| `Folds5x2_pp.xlsx`             | Original dataset (excluded via `.gitignore`)                               |
| `.gitignore`                   | Specifies files/folders to exclude from Git tracking (e.g., datasets, logs, model weights) |
| `requirements.txt`            | Python dependencies for reproducibility                                     |
| `LICENSE`                     | MIT License for open-source distribution                                   |
| `README.md`                   | Project documentation and portfolio presentation                           |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ArianJr/power-output-prediction-ann.git
cd power-output-prediction-ann
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch notebook
```bash
jupyter notebook power_output_prediction_ann.ipynb
```

---

## 📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute with proper attribution.

---

## 🙌 Acknowledgments
- UCI ML Repository for the dataset
- TensorFlow and Keras teams
- Open-source contributors and the GitHub community

---

## 🎯 Portfolio Value
This project showcases:
- Real-world regression modeling with deep learning
- Clean code, modular structure, and reproducible results
- Strong metric reporting and visual interpretability
- Ethical open-source practices and professional presentation

---
