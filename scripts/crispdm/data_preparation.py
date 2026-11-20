import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[2])  # paths below are relative to the project root
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

print("="*50)
print("CRISP-DM - DATA PREPARATION")
print("="*50)

# =========================================================
# LOAD DATA
# =========================================================
file_path = 'data/dataset.csv'
df = pd.read_csv(file_path)

print("\n[INFO] Dataset Loaded")

# =========================================================
# CLEANING
# =========================================================
print("\n[DATA CLEANING]")

# Missing value
df = df.fillna("Unknown")

# Normalisasi text
def normalize(x):
    return str(x).lower().strip()

df['Book_Author'] = df['Book_Author'].apply(normalize)
df['Publisher'] = df['Publisher'].apply(normalize)

print("Cleaning selesai")

# =========================================================
# ENCODING
# =========================================================
print("\n[ENCODING]")

le_author = LabelEncoder()
le_publisher = LabelEncoder()

df['Book_Author'] = le_author.fit_transform(df['Book_Author'])
df['Publisher'] = le_publisher.fit_transform(df['Publisher'])

print("Encoding selesai")

# =========================================================
# FEATURE ENGINEERING (sesuai train_model.py)
# =========================================================
print("\n[FEATURE ENGINEERING]")

df['book_age'] = 2025 - df['Year']
df['title_length'] = df['Book_Title'].apply(lambda x: len(str(x)))
df['title_word_count'] = df['Book_Title'].apply(lambda x: len(str(x).split()))
df['avg_word_length'] = df['Book_Title'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
)
df['decade'] = (df['book_age'] // 10) * 10

df['isbn_length'] = df['ISBN'].astype(str).str.replace('-', '', regex=False).str.len() if 'ISBN' in df.columns else 0
df['isbn_prefix'] = df['ISBN'].astype(str).str[:1].apply(
    lambda x: int(x) if str(x).isdigit() else -1
) if 'ISBN' in df.columns else -1

df['author_freq'] = df['Book_Author'].map(df['Book_Author'].value_counts()).fillna(1)
df['publisher_freq'] = df['Publisher'].map(df['Publisher'].value_counts()).fillna(1)
df['author_freq_log'] = np.log1p(df['author_freq'])
df['publisher_freq_log'] = np.log1p(df['publisher_freq'])

df['author_x_pub'] = df['Book_Author'] * df['Publisher']
df['age_x_author'] = df['book_age'] * df['author_freq_log']

author_map = df['Book_Author'].value_counts().to_dict()
publisher_map = df['Publisher'].value_counts().to_dict()

print("Feature engineering selesai")

# =========================================================
# VALIDASI DATA FINAL
# =========================================================
print("\n[VALIDASI DATA]")

required_cols = [
    'Book_Author',
    'Publisher',
    'book_age',
    'title_length',
    'title_word_count',
    'avg_word_length',
    'decade',
    'author_freq',
    'publisher_freq',
    'author_freq_log',
    'publisher_freq_log',
    'author_x_pub',
    'age_x_author',
]

for col in required_cols:
    if col not in df.columns:
        print(f"❌ {col} tidak ada!")
    else:
        print(f"✅ {col} OK")

# =========================================================
# SIMPAN DATA SIAP MODEL
# =========================================================
print("\n[SAVE PREPARED DATA]")

df.to_csv('data/clean_dataset.csv', index=False)

joblib.dump({
    'le_author': le_author,
    'le_publisher': le_publisher,
    'author_map': author_map,
    'publisher_map': publisher_map
}, 'models/encoder.pkl')

print("Data preparation selesai & disimpan")
print("="*50)