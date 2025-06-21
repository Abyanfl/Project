# Judul Project

## Repository Outline

1. README.md - Penjelasan umum tentang proyek prediksi attrition karyawan
2. notebook.ipynb - Notebook utama berisi eksplorasi data, preprocessing, visualisasi, modeling dan evaluasi
3. description.md - Deskripsi proyek
4. dataset/ - Berisi data mentah dan data hasil preprocessing

## Problem Background
Perusahaan menghadapi tantangan dalam mempertahankan karyawan berperforma tinggi. Tingginya angka attrition (keluar dari pekerjaan) dapat menyebabkan meningkatnya biaya rekrutmen dan hilangnya produktivitas. Oleh karena itu, diperlukan sistem prediktif yang dapat mengidentifikasi karyawan yang berisiko tinggi untuk keluar dari perusahaan, sehingga HR dapat melakukan intervensi lebih awal.

## Project Output
Model machine learning klasifikasi yang dapat memprediksi kemungkinan seorang karyawan akan keluar (attrition) berdasarkan karakteristik seperti umur, jarak dari rumah, departemen, jabatan, status pernikahan, jenis kelamin, dan lainnya. Output akhir juga mencakup visualisasi insight dan laporan evaluasi model.

## Data
Dataset yang digunakan adalah WA_Fn-UseC_-HR-Employee-Attrition.csv yang berisi 1.470 baris dan 35 kolom. Beberapa fitur utama meliputi:
Age, DistanceFromHome, Department, JobRole, MaritalStatus, Gender, MonthlyIncome, OverTime, dll.
Target variabel adalah Attrition dengan nilai 'Yes' atau 'No', yang akan dikonversi menjadi biner (Yes=1, No=0)
Tidak terdapat nilai kosong


## Method
Metode yang digunakan dalam proyek ini:
Exploratory Data Analysis (EDA)
Feature engineering (encoding, feature selection, binning)
Supervised learning (klasifikasi biner) dengan model:

KNN
SVM
Decision Tree
Random Forest
XGBoost

Evaluasi menggunakan akurasi, precision, recall, dan F1-score`

## Stacks
```
Bahasa pemrograman: Python 3
Tools: Jupyter Notebook
Libraries:
pandas, numpy, matplotlib, seaborn, scipy, statistics, pprint, warnings
plotly (express, graph_objects, subplots, offline)
scikit-learn (train_test_split, GridSearchCV, RandomizedSearchCV, preprocessing, pipelines, metrics, imputation, RandomForestClassifier, DecisionTreeClassifier, SVC)
xgboost (XGBClassifier)
imblearn (SMOTENC)
joblib

```

## Reference
```

Dataset IBM HR Analytics: WA_Fn-UseC_-HR-Employee-Attrition.csv (simulasi publik)
Dokumentasi Scikit-Learn: https://scikit-learn.org/stable/
Referensi teknik feature engineering dan EDA dari Kaggle & Medium


```