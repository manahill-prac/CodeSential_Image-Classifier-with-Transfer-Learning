# 🐶🐱 Cats vs Dogs Classifier (Transfer Learning with MobileNetV2)

This project builds a **custom image classifier** using **Transfer Learning** with **MobileNetV2** to classify images of cats and dogs.  
It applies **data augmentation, fine-tuning, evaluation metrics, and visualization** to deliver an internship-ready AI project.

---

## Dataset
- **Source:** TensorFlow Datasets → [`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)  
- **Size:** ~786MB, ~37,000 images (cats & dogs).  
- **Split:** 60% train, 20% validation, 20% test.  

---

##  Steps
1. **Import Libraries** – TensorFlow, TensorFlow Datasets, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.  
2. **Load Dataset** – Using `tfds.load` with train/val/test splits.  
3. **Preprocess & Augmentation** – Resize, Rescale, RandomFlip, RandomRotation, RandomZoom.  
4. **Build Model** – MobileNetV2 (pre-trained on ImageNet, without top layer) + custom Dense layers.  
5. **Train Model** – Initial training with frozen base layers, `EarlyStopping` & `ModelCheckpoint`.  
6. **Fine-Tuning** – Unfreeze MobileNetV2 for higher accuracy (low learning rate).  
7. **Visualize Training** – Accuracy & Loss curves with custom theme.  
8. **Evaluate on Test Data** – Accuracy, Loss, Precision, Recall, F1-score.  
9. **Confusion Matrix** – Styled heatmap.  
10. **Misclassified Images** – Display wrongly predicted samples.  
11. **Summary** – Key metrics + insights.

---

## Results
- **Best Validation Accuracy:** ~97–98%  
- **Test Accuracy:** ~97.85%  
- **Test Loss:** ~0.0597  

| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.9785  |
| Precision  | 0.9741  |
| Recall     | 0.9830  |
| F1-Score   | 0.9785  |

---

## How to Run

1. Clone this repository or download the notebook.
2. Install required dependencies.
3. Open the notebook in **Google Colab** (recommended) or Jupyter.
4. Run cells step by step to train & evaluate the model.

```

