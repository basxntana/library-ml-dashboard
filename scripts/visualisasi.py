import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # paths below are relative to the project root
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

print("📊 VISUALISASI DATA & MODEL")

# ================================
# LOAD DATASET
# ================================
data = pd.read_csv('data/dataset.csv')

# ================================
# STYLE
# ================================
sns.set_style("whitegrid")

# ================================
# 1. DISTRIBUSI KATEGORI
# ================================
plt.figure()
sns.countplot(x='kategori', data=data)
plt.title("Distribusi Kategori Buku")
plt.xlabel("Kategori (0 = Tidak Populer, 1 = Populer)")
plt.ylabel("Jumlah")
plt.savefig("output/distribusi_kategori.png")
plt.close()

# ================================
# 2. DISTRIBUSI JUMLAH PINJAM
# ================================
plt.figure()
sns.histplot(data['jumlah_pinjam'], bins=30, kde=True)
plt.title("Distribusi Jumlah Peminjaman")
plt.xlabel("Jumlah Pinjam")
plt.ylabel("Frekuensi")
plt.savefig("output/distribusi_pinjam.png")
plt.close()

# ================================
# 3. FEATURE IMPORTANCE
# ================================
model_data = joblib.load('models/model.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

importance = model.feature_importances_

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values(by='importance', ascending=False)

plt.figure()
sns.barplot(x='importance', y='feature', data=feat_imp)
plt.title("Feature Importance")
plt.savefig("output/feature_importance.png")
plt.close()

# ================================
# 4. CONFUSION MATRIX
# ================================
# Recreate data seperti di training
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = data.fillna("Unknown")

le_author = LabelEncoder()
le_publisher = LabelEncoder()

data['Book_Author'] = le_author.fit_transform(data['Book_Author'].astype(str))
data['Publisher'] = le_publisher.fit_transform(data['Publisher'].astype(str))

X = data.drop(columns=['kategori', 'Book_Title'])
y = data['kategori']

# transform log
if 'jumlah_pinjam' in X.columns:
    X['jumlah_pinjam_log'] = np.log1p(X['jumlah_pinjam'])
    X = X.drop(columns=['jumlah_pinjam'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output/confusion_matrix.png")
plt.close()

print("\n✅ Semua visualisasi berhasil dibuat di folder 'output/'")