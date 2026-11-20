import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[2])  # paths below are relative to the project root
import pandas as pd

print("="*50)
print("CRISP-DM - DATA UNDERSTANDING")
print("="*50)

# =========================================================
# LOAD DATA
# =========================================================
file_path = 'data/dataset.csv'
df = pd.read_csv(file_path)

print("\n[INFO] Dataset berhasil dimuat")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")

# =========================================================
# STRUKTUR DATA
# =========================================================
print("\n[STRUKTUR DATA]")
print(df.info())

# =========================================================
# PREVIEW DATA
# =========================================================
print("\n[5 DATA TERATAS]")
print(df.head())

# =========================================================
# DESKRIPSI STATISTIK
# =========================================================
print("\n[STATISTIK NUMERIK]")
print(df.describe())

# =========================================================
# CEK MISSING VALUE
# =========================================================
print("\n[CEK MISSING VALUE]")
missing = df.isnull().sum()
print(missing)

# =========================================================
# DISTRIBUSI TARGET
# =========================================================
print("\n[DISTRIBUSI TARGET]")
if 'kategori' in df.columns:
    print(df['kategori'].value_counts())
else:
    print("Kolom 'kategori' tidak ditemukan!")

# =========================================================
# ANALISIS SEDERHANA
# =========================================================

# TOP 5 BUKU PALING DIPINJAM
if 'Book_Title' in df.columns and 'jumlah_pinjam' in df.columns:
    top_books = df.groupby('Book_Title')['jumlah_pinjam'] \
        .sum().sort_values(ascending=False).head(5)

    print("\n[TOP 5 BUKU PALING DIPINJAM]")
    print(top_books)

# TOP AUTHOR
if 'Book_Author' in df.columns:
    top_author = df['Book_Author'].value_counts().head(5)

    print("\n[TOP 5 AUTHOR TERBANYAK]")
    print(top_author)

# TOP PUBLISHER
if 'Publisher' in df.columns:
    top_publisher = df['Publisher'].value_counts().head(5)

    print("\n[TOP 5 PUBLISHER TERBANYAK]")
    print(top_publisher)

# =========================================================
# CEK DUPLIKAT
# =========================================================
duplicates = df.duplicated().sum()
print(f"\n[DATA DUPLIKAT]: {duplicates}")

# =========================================================
# CEK NILAI UNIK
# =========================================================
print("\n[NILAI UNIK PER KOLOM]")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unik")

# =========================================================
# VALIDASI DATA
# =========================================================
required_cols = ['Book_Title','Book_Author','Publisher','jumlah_pinjam','Year','kategori']

print("\n[VALIDASI KOLOM]")
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Kolom {col} TIDAK ADA")
    else:
        print(f"✅ Kolom {col} tersedia")

print("\nData Understanding Selesai!")
print("="*50)