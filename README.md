# Adult Income Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

The project implements five classification algorithms and evaluates their performance using six evaluation metrics. It also examines the effect of Random Oversampling on handling class imbalance.

## 2. Dataset Description

- **Dataset Name:** Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Original Instances:** 48,842
- **Features:** 14
- **Experimental Sample Size:** 12,000
- **Target Variable:** Income
- **Classification Type:** Binary Classification

**Target Classes:**
- `<=50K`
- `>50K`

**Reason for Dataset Selection:**

The Adult Income dataset was selected because it contains both numerical and categorical features, has sufficient instances and features to satisfy the assignment requirements, and provides an opportunity to study missing values and class imbalance in a real-world classification problem.

**Feature Types:**

The dataset contains numerical features such as age, education number, capital gain, capital loss and hours per week, along with categorical features such as workclass, education, marital status, occupation, relationship, race, sex and native country.

**Class Distribution:**

After duplicate removal:
- Income <=50K: 37,109 records (76.06%)
- Income >50K: 11,681 records (23.94%)

The dataset was examined for missing values. The following missing values were identified:
Feature	Missing Values
Occupation	2,805
Workclass	2,795
Native Country	856

Missing numerical values were handled using median imputation, while missing categorical values were handled using most-frequent imputation.

The dataset is imbalanced, with the <=50K class forming the majority.

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Identified missing values and replaced missing-value markers with null values.
2. Removed duplicate records.
3. Applied median imputation for numerical features.
4. Applied most-frequent imputation for categorical features.
5. Applied One-Hot Encoding to categorical features.
6. Applied Standard Scaling to numerical features.
7. Performed an 80:20 stratified train-test split.
8. Used random state 42 to ensure reproducibility.
9. Fitted preprocessing transformations only on training data using Scikit-learn pipelines.

A sample of 12,000 records was used for the experiments.

## 4. GitHub Repository Link

**Repository:** https://github.com/2026ct05035-del/adult-income-classification-ml

The repository contains the Jupyter Notebook, test dataset, requirements file, saved model directory and output files.

## 5. Machine Learning Models Used

The following five classification algorithms were implemented:

### 5.1 Logistic Regression
- Maximum iterations: 1500
- Solver: liblinear
- Random state: 42

### 5.2 Decision Tree Classifier
- Maximum depth: 12
- Minimum samples per leaf: 5
- Random state: 42

### 5.3 K-Nearest Neighbors (kNN)
- Number of neighbors: 7
- Weights: distance-based

### 5.4 Gaussian Naive Bayes
- Classifier: GaussianNB
- Used for probabilistic classification.

### 5.5 Random Forest Classifier (Ensemble)
- Number of estimators: 250
- Minimum samples per leaf: 2
- Class weight: balanced_subsample
- Random state: 42
Note:The original Random Forest experiment used class_weight="balanced_subsample". The oversampled experiment used Random Oversampling to address class imbalance, with the corresponding model configuration.

## 6. Model Evaluation and Comparison

The models were evaluated using the following six metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Performance Comparison Table

The following results are from the original experiment without Random Oversampling.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8554 | 0.9092 | 0.7415 | 0.6087 | 0.6686 | 0.5819 |
| Decision Tree | 0.8450 | 0.8735 | 0.7101 | 0.5965 | 0.6484 | 0.5534 |
| kNN | 0.8287 | 0.8732 | 0.6608 | 0.5861 | 0.6212 | 0.5126 |
| Naive Bayes | 0.3442 | 0.6194 | 0.2643 | 0.9739 | 0.4157 | 0.1589 |
| Random Forest (Ensemble) | 0.8383 | 0.9152 | 0.6238 | 0.8191 | 0.7083 | 0.6099 |

### Performance Comparison – Random Oversampling

The following results were obtained after applying Random Oversampling to the training data.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8050 | 0.9057 | 0.5617 | 0.8470 | 0.6755 | 0.5676 |
| Decision Tree | 0.8033 | 0.8559 | 0.5640 | 0.7896 | 0.6580 | 0.5399 |
| kNN | 0.7796 | 0.8547 | 0.5272 | 0.7757 | 0.6277 | 0.4972 |
| Naive Bayes | 0.3450 | 0.6162 | 0.2647 | 0.9757 | 0.4165 | 0.1619 |
| Random Forest (Ensemble) | 0.8387 | 0.9123 | 0.6309 | 0.7878 | 0.7007 | 0.5990 |

### Graphical Representation

The model evaluation results are represented through visualizations, including:

- Comparison of all six evaluation metrics across the five models.
- Heatmap showing the performance of each classification model.

The generated visualization files are available in the `outputs/` directory.

## 7. Observations on Model Performance

### Logistic Regression
Achieved the highest accuracy (85.54%) and precision (74.15%). It provides good overall classification performance but has comparatively lower recall than Random Forest.

### Decision Tree
Achieved an accuracy of 84.50%. It provides reasonable classification performance but has lower AUC and F1 scores compared with Random Forest.

### K-Nearest Neighbors (kNN)
Achieved an accuracy of 82.87%. Its performance is moderate, with comparatively lower F1 and MCC scores.

### Naive Bayes
Achieved very high recall (97.39%) but very low accuracy (34.42%) and MCC (0.1589). It identifies most positive-class instances but produces many false positives.

### Random Forest (Ensemble)
Achieved the highest AUC (0.9152), F1 Score (0.7083) and MCC (0.6099). It provides the strongest overall balance among the evaluated models.

### Overall Winner for the Dataset

**Random Forest Classifier**

Random Forest is considered the overall winner because it achieved the highest AUC, F1 Score and MCC. These metrics indicate good discrimination ability and a better balance between precision and recall compared with the other evaluated classifiers.

## 8. Random Oversampling Experiment

A separate experiment was conducted using Random Oversampling to address class imbalance.

The results showed that oversampling improved recall for some classifiers, particularly Logistic Regression. However, it did not improve every evaluation metric for every model.

For Random Forest, the original experiment achieved better AUC, recall, F1 and MCC than the oversampled experiment. This demonstrates that oversampling does not necessarily improve overall model performance.

## 9. Project Structure

```text
adult-income-classification-ml/
│
├── 2026CT05035_ML2.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   └── oversampled/
│
└── outputs/
```

## 10. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## 11. Execution Instructions

1. Clone the GitHub repository.
2. Install the required dependencies.
3. Open the Jupyter Notebook.
4. Execute all notebook cells sequentially.

**Installation:**

```bash
pip install -r requirements.txt
```

**Clone the repository:**

```bash
git clone https://github.com/2026ct05035-del/adult-income-classification-ml.git
cd adult-income-classification-ml
```

**Open Jupyter Notebook:**

```bash
jupyter notebook
```

## 12. Conclusion

This project implemented and evaluated five machine learning classification algorithms using the UCI Adult Income dataset.

The experiments demonstrated differences in model performance across accuracy, AUC, precision, recall, F1 and MCC. Random Forest achieved the best overall performance in the original experiment, while the separate oversampling experiment highlighted the impact of class imbalance handling on classification results.

The project provided practical experience in data preprocessing, classification model implementation, performance evaluation and comparison of machine learning algorithms.

---

**Course:** Machine Learning  
**Student ID:** 2026CT05035
