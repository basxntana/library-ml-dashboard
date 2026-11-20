import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # paths below are relative to the project root
import pandas as pd
import numpy as np

print("="*60)
print("CRISP-DM - FAST DATA PREPARATION")
print("="*60)

# =========================
# LOAD DATA
# =========================
books = pd.read_csv('data/Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('data/Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

books.columns = ['ISBN', 'Book_Title', 'Book_Author', 'Year', 'Publisher']
ratings.columns = ['User_ID', 'ISBN', 'Book_Rating']

# =========================
# MERGE
# =========================
data = pd.merge(ratings, books, on='ISBN')
print("Total raw:", len(data))

# =========================
# CLEAN
# =========================
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Year'] = data['Year'].fillna(data['Year'].median())

data = data.fillna("unknown")

data['Book_Author'] = data['Book_Author'].str.lower().str.strip()
data['Publisher'] = data['Publisher'].str.lower().str.strip()

# =========================
# AGGREGATION
# =========================
df = data.groupby('ISBN').agg({
    'Book_Title': 'first',
    'Book_Author': 'first',
    'Publisher': 'first',
    'Year': 'first'
}).reset_index()

df['jumlah_pinjam'] = data.groupby('ISBN').size().values

# =========================
# FEATURE ENGINEERING (FAST)
# =========================
df['book_age'] = 2025 - df['Year']
df['title_length'] = df['Book_Title'].astype(str).str.len()

# 🔥 FAST SIGNAL (NO LEAKAGE)
df['author_freq'] = df['Book_Author'].map(df['Book_Author'].value_counts())
df['publisher_freq'] = df['Publisher'].map(df['Publisher'].value_counts())

# =========================
# LABEL (Q75 — konsisten dengan train_model.py)
# =========================
threshold = df['jumlah_pinjam'].quantile(0.75)
df['kategori'] = (df['jumlah_pinjam'] > threshold).astype(int)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/dataset.csv", index=False)

print("\nDataset ready:", df.shape)
print(df['kategori'].value_counts())

# =========================
# EXPORT SATU DATASET VISUALIZATION
# =========================
print("\n" + "="*60)
print("EXPORT SINGLE CSV FOR VISUALIZATION")
print("="*60)

# =========================
# 1. DISTRIBUSI KATEGORI
# =========================
kategori_dist = df['kategori'].value_counts().reset_index()
kategori_dist.columns = ['value', 'jumlah']
kategori_dist['tipe'] = 'kategori'

# =========================
# 2. TOP AUTHOR
# =========================
top_author = df['Book_Author'].value_counts().head(10).reset_index()
top_author.columns = ['value', 'jumlah']
top_author['tipe'] = 'author'

# =========================
# 3. TOP PUBLISHER
# =========================
top_pub = df['Publisher'].value_counts().head(10).reset_index()
top_pub.columns = ['value', 'jumlah']
top_pub['tipe'] = 'publisher'

# =========================
# 4. BOOK AGE DISTRIBUTION
# =========================
umur_dist = df['book_age'].value_counts().reset_index()
umur_dist.columns = ['value', 'jumlah']
umur_dist['tipe'] = 'book_age'

# =========================
# 5. JUMLAH PINJAM (BINNING BIAR RAPI)
# =========================
df['pinjam_bin'] = pd.cut(df['jumlah_pinjam'], bins=10)

pinjam_dist = df['pinjam_bin'].value_counts().reset_index()
pinjam_dist.columns = ['value', 'jumlah']
pinjam_dist['tipe'] = 'jumlah_pinjam'

# =========================
# GABUNG SEMUA
# =========================
viz_df = pd.concat([
    kategori_dist,
    top_author,
    top_pub,
    umur_dist,
    pinjam_dist
], ignore_index=True)

# =========================
# SAVE
# =========================
viz_df.to_csv("data/dataset_viz.csv", index=False)

print("\n✅ dataset_viz.csv berhasil dibuat!")
print(viz_df.head())