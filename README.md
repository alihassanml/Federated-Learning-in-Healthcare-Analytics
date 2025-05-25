# Federated Learning in Healthcare Analytics 🧠🏥  
A Federated Learning system built on the NIH Chest X-ray Dataset to perform multi-label classification of thoracic diseases while preserving patient privacy. Implemented using TensorFlow and TensorFlow Federated (TFF).

---

## 📌 Project Goals
- Detect 14 thoracic diseases (e.g., Pneumonia, Effusion, Cardiomegaly) using chest X-rays.
- Simulate Federated Learning across multiple patient devices.
- Preserve data privacy by training models locally and aggregating weights centrally.

---

## 🗂 Dataset

**Source:** [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

- 112,120 frontal chest X-ray images
- 30,805 unique patients
- 14 disease labels + “No Finding”
- NLP-extracted labels (90%+ accurate)
- Image size: 1024x1024, resized to 224x224 for training

| Class Labels |
|--------------|
| Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia |

---

## ⚙️ Tech Stack

- Python 3.9
- TensorFlow 2.x
- TensorFlow Federated
- Scikit-learn, Pandas, NumPy
- Kaggle GPU Notebooks (for model training)

---

## 🚀 How to Run (Kaggle or Locally)

1. **Clone the repo**
```bash
git clone https://github.com/alihassanml/Federated-Learning-in-Healthcare-Analytics-.git
cd Federated-Learning-in-Healthcare-Analytics-
````

2. **Upload and Extract Dataset**
   Upload `images_001.zip` to `images_012.zip` and `Data_Entry_2017.csv` in your Kaggle notebook or local environment.

3. **Run Main Notebook**
   Train the baseline CNN:

```python
python train_tensorflow_model.py
```

4. **Run Federated Learning Simulation**
   Simulate multiple patient-wise clients:

```python
python federated_training.py
```

---

## 🧠 Model Architecture

**Baseline CNN (Transfer Learning)**

* DenseNet121 (ImageNet pre-trained)
* Global Average Pooling
* Output: 14 sigmoid-activated neurons (multi-label)

**Federated Model**

* Conv2D + MaxPooling
* Flatten + Dense (14 outputs)

---

## 📉 Metrics

* Multi-label Binary Accuracy
* Binary Crossentropy Loss
* Federated Learning rounds tracked by `tff.learning.build_federated_averaging_process`

---

## 📂 Repository Structure

```bash
Federated-Learning-in-Healthcare-Analytics-/
├── train_tensorflow_model.py       # TensorFlow model training
├── federated_training.py           # Federated Learning with TFF
├── requirements.txt
├── README.md
└── dataset/
    ├── Data_Entry_2017.csv
    └── images/
```

---

## 📚 Citation

```
@article{wang2017chestxray8,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={IEEE CVPR},
  year={2017}
}
```

---

## 🤝 Acknowledgements

This project uses the publicly available NIH ChestX-ray8 dataset made available by the NIH Clinical Center and the National Library of Medicine.

---

## 📧 Contact

**Ali Hassan**
`alihassanbscs99@gmail.com`
Student - BS Computer Science, LGU
