# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    classification_report, confusion_matrix, accuracy_score
)

# 2. Load Dataset
data = pd.read_csv('student-mat.csv', sep=',')
print("=== 5 Data Pertama ===")
print(data.head())
print("\nData ini memuat informasi siswa termasuk jam belajar, kegagalan, dan nilai G1-G3.")

# 3. EDA - Exploratory Data Analysis
print("\n=== Info Data ===")
print(data.info())
print("\nDataset terdiri dari", data.shape[0], "siswa dengan", data.shape[1], "fitur.")

print("\n=== Statistik Deskriptif ===")
print(data.describe())

print("\n=== Cek Missing Value ===")
print(data.isnull().sum())
print("\nTidak ditemukan missing value, sehingga data aman untuk dianalisis.")

# Visualisasi Distribusi Nilai
plt.figure(figsize=(12,4))
for i, col in enumerate(['G1', 'G2', 'G3']):
    plt.subplot(1,3,i+1)
    sns.histplot(data[col], kde=True, color='skyblue')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()
print("\nVisualisasi di atas menunjukkan sebaran nilai siswa dari ujian pertama (G1) hingga akhir (G3).")

# 4. Buat Label Lulus (G3 >=10)
data['pass'] = np.where(data['G3'] >= 10, 1, 0)
print("\nKolom 'pass' baru dibuat: 1 = Lulus (G3 >= 10), 0 = Tidak Lulus.")

# 5. Fitur yang Dipilih
features = ['studytime', 'failures', 'G1', 'G2']
X = data[features]

# Target
y_reg = data['G3']
y_clf = data['pass']

# 6. Split Data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# ============================
# 🔥 REGRESI LINIER
# ============================
print("\n=== Regresi Linier (Prediksi Nilai G3) ===")
lin_model = LinearRegression()
lin_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lin_model.predict(X_test_reg)

print("MSE (Semakin kecil semakin baik):", mean_squared_error(y_test_reg, y_pred_reg))
print("R2 Score (Semakin dekat ke 1 semakin baik):", r2_score(y_test_reg, y_pred_reg))

# Visualisasi Hasil
plt.scatter(y_test_reg, y_pred_reg, color='green')
plt.xlabel('Nilai Aktual G3')
plt.ylabel('Nilai Prediksi G3')
plt.title('Regresi Linier: Prediksi Nilai Akhir G3')
plt.show()
print("\nGrafik scatter menunjukkan kecocokan prediksi (titik hijau) terhadap nilai asli.")

# ============================
# 🔥 REGRESI LOGISTIK
# ============================
print("\n=== Regresi Logistik (Prediksi Lulus / Tidak) ===")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_clf, y_train_clf)

y_pred_clf = log_model.predict(X_test_clf)

print("Akurasi Model Logistik:", accuracy_score(y_test_clf, y_pred_clf))
print("\nLaporan Klasifikasi:\n", classification_report(y_test_clf, y_pred_clf))

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Lulus', 'Lulus'], yticklabels=['Tidak Lulus', 'Lulus'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix - Regresi Logistik')
plt.show()
print("\nMatriks di atas menunjukkan jumlah prediksi benar/salah untuk siswa lulus atau tidak.")

# ============================
# 🔥 Tambahan: Random Forest & SVM
# ============================

# Random Forest
print("\n=== Random Forest Classifier ===")
rf_model = RandomForestClassifier()
rf_model.fit(X_train_clf, y_train_clf)
y_pred_rf = rf_model.predict(X_test_clf)
print("Akurasi Random Forest:", accuracy_score(y_test_clf, y_pred_rf))
print("\nRandom Forest biasanya lebih kuat karena menggunakan banyak pohon keputusan.")

# SVM
print("\n=== Support Vector Machine (SVM) ===")
svm_model = SVC()
svm_model.fit(X_train_clf, y_train_clf)
y_pred_svm = svm_model.predict(X_test_clf)
print("Akurasi SVM:", accuracy_score(y_test_clf, y_pred_svm))
print("\nSVM efektif untuk data dengan batas kelas yang jelas.")

# ============================
# 🔥 Cross Validation (Logistik)
# ============================
cv_scores = cross_val_score(log_model, X, y_clf, cv=5, scoring='accuracy')
print("\nAkurasi Cross-Validation (5-fold):", cv_scores)
print("Rata-rata Akurasi CV:", np.mean(cv_scores))
print("\nCross-validation membantu memastikan model stabil di berbagai subset data.")

# ============================
# 🔥 Feature Importance (Logistik)
# ============================
importance = log_model.coef_[0]
feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

feature_importance.plot(kind='barh', color='coral')
plt.xlabel('Pengaruh terhadap Kelulusan (Koefisien)')
plt.title('Faktor yang Mempengaruhi Kelulusan Siswa')
plt.show()
print("\nGrafik ini menunjukkan faktor dominan kelulusan: G2 paling berpengaruh, diikuti G1, studytime, dan failures.")
