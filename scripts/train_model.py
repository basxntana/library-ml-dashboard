import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # paths below are relative to the project root
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# =========================================================
# CRISP-DM - RANDOM FOREST MAXIMIZED
# =========================================================
print("=" * 60)
print("  CRISP-DM - RANDOM FOREST")
print("=" * 60)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/dataset.csv")
df = df.fillna("unknown")

print(f"\n[1] Total Data    : {len(df):,} baris")
print(f"    Total Kolom   : {df.shape[1]} kolom")

# =========================================================
# 2. CLEAN TEXT
# =========================================================
df['Book_Author'] = df['Book_Author'].astype(str).str.lower().str.strip()
df['Publisher']   = df['Publisher'].astype(str).str.lower().str.strip()
df['Book_Title']  = df['Book_Title'].astype(str).str.lower().str.strip()
df['ISBN']        = df['ISBN'].astype(str).str.strip()

# =========================================================
# 3. LABEL — NATURAL DISTRIBUTION (Q75)
# =========================================================
threshold = df['jumlah_pinjam'].quantile(0.75)
df['kategori'] = (df['jumlah_pinjam'] > threshold).astype(int)

total      = len(df)
n_populer  = df['kategori'].sum()
n_tidak    = total - n_populer

print(f"\n[2] Threshold Label (Q75) : {threshold}")
print(f"    Tidak Populer (0)     : {n_tidak:,} ({n_tidak/total*100:.1f}%)")
print(f"    Populer (1)           : {n_populer:,} ({n_populer/total*100:.1f}%)")

# =========================================================
# 4. ENCODING
# =========================================================
le_author    = LabelEncoder()
le_publisher = LabelEncoder()

df['Book_Author'] = le_author.fit_transform(df['Book_Author'])
df['Publisher']   = le_publisher.fit_transform(df['Publisher'])

# =========================================================
# 5. FEATURE ENGINEERING — MAKSIMAL
# =========================================================

# Fitur dasar
df['book_age']         = 2025 - pd.to_numeric(df['Year'], errors='coerce').fillna(2025)
df['title_length']     = df['Book_Title'].str.len()
df['title_word_count'] = df['Book_Title'].str.split().str.len()

#  FITUR BARU: Ekstrak dari ISBN
# ISBN digit pertama = kode bahasa/negara (0/1 = Inggris, 2 = Prancis, dst)
df['isbn_length']  = df['ISBN'].str.replace('-', '').str.len()  # 10 vs 13
df['isbn_prefix']  = df['ISBN'].str[:1].apply(
    lambda x: int(x) if x.isdigit() else -1
)

#  FITUR BARU: Title complexity
df['avg_word_length'] = df['Book_Title'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
)

#  FITUR BARU: Era buku (dekade)
df['decade'] = (df['book_age'] // 10) * 10

# =========================================================
# 6. SPLIT DULU — BARU FREQUENCY FEATURES (anti-leakage)
# =========================================================
y = df['kategori']

X_train_temp, X_test_temp, y_train, y_test = train_test_split(
    df, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Frequency features dari train saja
author_freq_map    = X_train_temp['Book_Author'].value_counts().to_dict()
publisher_freq_map = X_train_temp['Publisher'].value_counts().to_dict()

X_train_temp = X_train_temp.copy()
X_test_temp  = X_test_temp.copy()

X_train_temp['author_freq']    = X_train_temp['Book_Author'].map(author_freq_map).fillna(1)
X_train_temp['publisher_freq'] = X_train_temp['Publisher'].map(publisher_freq_map).fillna(1)
X_test_temp['author_freq']     = X_test_temp['Book_Author'].map(author_freq_map).fillna(1)
X_test_temp['publisher_freq']  = X_test_temp['Publisher'].map(publisher_freq_map).fillna(1)

# Log frequency (lebih stabil untuk skewed data)
X_train_temp['author_freq_log']    = np.log1p(X_train_temp['author_freq'])
X_train_temp['publisher_freq_log'] = np.log1p(X_train_temp['publisher_freq'])
X_test_temp['author_freq_log']     = np.log1p(X_test_temp['author_freq'])
X_test_temp['publisher_freq_log']  = np.log1p(X_test_temp['publisher_freq'])

# Interaction features
X_train_temp['author_x_pub']     = X_train_temp['Book_Author'] * X_train_temp['Publisher']
X_test_temp['author_x_pub']      = X_test_temp['Book_Author']  * X_test_temp['Publisher']

X_train_temp['age_x_author']     = X_train_temp['book_age'] * X_train_temp['author_freq_log']
X_test_temp['age_x_author']      = X_test_temp['book_age']  * X_test_temp['author_freq_log']

# =========================================================
# 7. FINAL FEATURES
# =========================================================
features = [
    'Book_Author',
    'Publisher',
    'book_age',
    'decade',
    'title_length',
    'title_word_count',
    'avg_word_length',
    'isbn_length',
    'isbn_prefix',
    'author_freq',
    'publisher_freq',
    'author_freq_log',
    'publisher_freq_log',
    'author_x_pub',
    'age_x_author',
]

X_train = X_train_temp[features]
X_test  = X_test_temp[features]

print(f"\n[3] Total Fitur   : {len(features)}")
print(f"    Fitur         : {features}")

# =========================================================
# 8. CROSS VALIDATION — 5-FOLD + SMOTE DI DALAM FOLD
# =========================================================
print(f"\n[4] Stratified 5-Fold Cross Validation...")

# Penting:
# SMOTE dilakukan DI DALAM masing-masing training fold,
# bukan sebelum Cross Validation.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []
oof_proba = np.zeros(len(X_train))

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train), start=1
):
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val   = X_train.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val   = y_train.iloc[val_idx]

    # Pipeline: SMOTE hanya diterapkan pada training fold
    fold_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42
        ))
    ])

    fold_pipeline.fit(X_fold_train, y_fold_train)

    # Probabilitas validation fold untuk OOF prediction
    y_proba_fold = fold_pipeline.predict_proba(X_fold_val)[:, 1]
    y_pred_fold = (y_proba_fold >= 0.50).astype(int)

    # Simpan OOF probability untuk Threshold Tuning
    oof_proba[val_idx] = y_proba_fold

    # Metrik tiap fold
    fold_acc = accuracy_score(y_fold_val, y_pred_fold)
    fold_prec = precision_score(
        y_fold_val, y_pred_fold,
        average='weighted', zero_division=0
    )
    fold_rec = recall_score(
        y_fold_val, y_pred_fold,
        average='weighted', zero_division=0
    )
    fold_f1 = f1_score(
        y_fold_val, y_pred_fold,
        average='weighted', zero_division=0
    )

    fold_results.append({
        'Fold': fold,
        'Accuracy': fold_acc,
        'Precision': fold_prec,
        'Recall': fold_rec,
        'F1-Score': fold_f1
    })

# Tabel hasil 5-Fold
cv_results = pd.DataFrame(fold_results)

cv_mean = {
    'Fold': 'Rata-rata',
    'Accuracy': cv_results['Accuracy'].mean(),
    'Precision': cv_results['Precision'].mean(),
    'Recall': cv_results['Recall'].mean(),
    'F1-Score': cv_results['F1-Score'].mean()
}

cv_results = pd.concat(
    [cv_results, pd.DataFrame([cv_mean])],
    ignore_index=True
)

print("\nHasil 5-Fold Cross Validation:")
print(
    cv_results.to_string(
        index=False,
        formatters={
            'Accuracy': '{:.4f}'.format,
            'Precision': '{:.4f}'.format,
            'Recall': '{:.4f}'.format,
            'F1-Score': '{:.4f}'.format
        }
    )
)

print(f"\nMean CV F1     : {cv_mean['F1-Score'] * 100:.2f}%")
print(f"Std CV F1      : {cv_results.iloc[:5]['F1-Score'].std() * 100:.2f}%")

# =========================================================
# 9. SMOTE — Oversample seluruh TRAINING DATA
#    Dilakukan setelah Cross Validation selesai
# =========================================================
print(f"\n[5] SMOTE pada seluruh data training...")
print(
    f"    Sebelum -> 0: {y_train.value_counts()[0]:,} | "
    f"1: {y_train.value_counts()[1]:,}"
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(
    f"    Sesudah -> 0: {pd.Series(y_train_sm).value_counts()[0]:,} | "
    f"1: {pd.Series(y_train_sm).value_counts()[1]:,}"
)

# =========================================================
# 10. MODEL — PARAMETER OPTIMAL
# =========================================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)

# =========================================================
# 11. TRAIN MODEL FINAL
# =========================================================
print(f"\n[6] Training model final...")
model.fit(X_train_sm, y_train_sm)
print(f"    Training selesai.")

# =========================================================
# 12. THRESHOLD TUNING — OOF, BUKAN TEST SET
# =========================================================
# Threshold dipilih berdasarkan Out-of-Fold (OOF) prediction
# agar test set tetap benar-benar belum digunakan.
best_threshold = 0.50
best_f1_macro = -1

for thresh in np.arange(0.25, 0.70, 0.05):
    y_oof_pred = (oof_proba >= thresh).astype(int)
    f1_macro_t = f1_score(
        y_train,
        y_oof_pred,
        average='macro',
        zero_division=0
    )

    if f1_macro_t > best_f1_macro:
        best_f1_macro = f1_macro_t
        best_threshold = thresh

print(f"\n[7] Threshold Tuning (berdasarkan OOF)")
print(f"    Threshold terbaik : {best_threshold:.2f}")
print(f"    OOF F1 Macro      : {best_f1_macro * 100:.2f}%")

# =========================================================
# 13. FINAL EVALUATION — TEST SET ASLI
# =========================================================
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
f1_m  = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_0  = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_1  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
auc   = roc_auc_score(y_test, y_proba)
cm    = confusion_matrix(y_test, y_pred)

print("\n")
print("=" * 60)
print("  HASIL EVALUASI AKHIR")
print("=" * 60)
print(f"  Threshold Terbaik  : {best_threshold:.2f}")
print(f"  Accuracy           : {acc*100:.2f}%")
print(f"  Precision (w)      : {prec*100:.2f}%")
print(f"  Recall (w)         : {rec*100:.2f}%")
print(f"  F1 Score (weighted): {f1_w*100:.2f}%")
print(f"  F1 Score (macro)   : {f1_m*100:.2f}%")
print(f"  F1 Kelas 0         : {f1_0*100:.2f}%")
print(f"  F1 Kelas 1 (populer): {f1_1*100:.2f}%")
print(f"  AUC-ROC            : {auc:.4f}")
print("-" * 60)
print(f"\n  Confusion Matrix:")
print(f"  {'':>20} Pred 0    Pred 1")
print(f"  {'Aktual 0 (Tidak Populer)':>24}  {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"  {'Aktual 1 (Populer)':>24}  {cm[1][0]:>6}    {cm[1][1]:>6}")
print("-" * 60)
print(f"\n  Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Tidak Populer', 'Populer']
))

# =========================================================
# 13B. SIMPAN HASIL CROSS VALIDATION UNTUK SKRIPSI
# =========================================================
cv_results.to_csv(
    "output/hasil_5_fold_cross_validation.csv",
    index=False
)

print("\nTabel 5-Fold tersimpan -> output/hasil_5_fold_cross_validation.csv")

# =========================================================
# 14. FEATURE IMPORTANCE
# =========================================================
print("=" * 60)
print("  FEATURE IMPORTANCE")
print("=" * 60)
importance_pairs = sorted(
    zip(features, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for feat, imp in importance_pairs:
    bar = "#" * int(imp * 60)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# =========================================================
# 15. RINGKASAN UNTUK SKRIPSI
# =========================================================
print("\n")
print("=" * 60)
print("  RINGKASAN UNTUK PENULISAN ILMIAH")
print("=" * 60)
print(f"  Dataset            : {len(df):,} data buku")
print(f"  Threshold Populer  : jumlah_pinjam > {threshold:.0f} (Q75)")
print(f"  Rasio Kelas        : {n_tidak/total*100:.0f}% tidak populer, {n_populer/total*100:.0f}% populer")
print(f"  Metode Balancing   : SMOTE (Synthetic Minority Oversampling)")
print(f"  Algoritma          : Random Forest ({model.n_estimators} trees, depth {model.max_depth})")
print(f"  Split Data         : 80% train, 20% test (stratified)")
print(f"  Cross Validation   : 5-fold Stratified CV")
print(f"  Mean CV Accuracy   : {cv_mean['Accuracy']*100:.2f}%")
print(f"  Mean CV Precision  : {cv_mean['Precision']*100:.2f}%")
print(f"  Mean CV Recall     : {cv_mean['Recall']*100:.2f}%")
print(f"  Mean CV F1-Score   : {cv_mean['F1-Score']*100:.2f}%")
print(f"  Threshold Optimal  : {best_threshold:.2f}")
print(f"  Akurasi            : {acc*100:.2f}%")
print(f"  F1 Macro           : {f1_m*100:.2f}%")
print(f"  AUC-ROC            : {auc:.4f}")
print("=" * 60)

# =========================================================
# 16. SIMPAN MODEL
# =========================================================
import joblib

model_bundle = {
    'model': model,
    'features': features,
    'feature_names': features,
    'le_author': le_author,
    'le_publisher': le_publisher,
    'author_freq_map': author_freq_map,
    'publisher_freq_map': publisher_freq_map,
    'best_threshold': best_threshold,
    'cv_results': cv_results.to_dict(orient='records'),
}

import tempfile, os

model_bundle['eval_metrics'] = {
    'accuracy': round(acc, 4),
    'precision': round(prec, 4),
    'recall': round(rec, 4),
    'f1': round(f1_w, 4),
    'f1_macro': round(f1_m, 4),
    'auc_roc': round(auc, 4),
    'confusion_matrix': cm.tolist(),
}

save_path = 'models/model.pkl'
try:
    joblib.dump(model_bundle, save_path)
except OSError:
    save_path = os.path.join(tempfile.gettempdir(), 'library_ml_model.pkl')
    joblib.dump(model_bundle, save_path)
print(f"\n MODEL TERSIMPAN -> {save_path}")
print(f"  best_threshold: {best_threshold:.2f}")
print(f"  features: {len(features)}")
print("=" * 60)