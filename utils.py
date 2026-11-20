import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def normalize(x):
    return str(x).lower().strip()


def safe_encode(le, value):
    if le is None:
        return 0
    value = normalize(value)
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


def build_single_input(req):
    return {
        'jumlah_pinjam': float(req.get('jumlah_pinjam', 0)),
        'year': float(req.get('year', 2000)),
        'title': str(req.get('title', "Unknown")),
        'author': str(req.get('author', "Unknown")),
        'publisher': str(req.get('publisher', "Unknown"))
    }


def build_model_input(df, feature_names, le_author=None, le_publisher=None,
                       author_freq_map=None, publisher_freq_map=None):
    df = df.copy()
    df = df.fillna("unknown")

    if 'Book_Title' not in df.columns:
        df['Book_Title'] = "unknown"
    if 'Book_Author' not in df.columns:
        df['Book_Author'] = "unknown"
    if 'Publisher' not in df.columns:
        df['Publisher'] = "unknown"
    if 'Year' not in df.columns:
        df['Year'] = 2025
    if 'ISBN' not in df.columns:
        df['ISBN'] = "0"
    if 'jumlah_pinjam' not in df.columns:
        df['jumlah_pinjam'] = 0

    df['Book_Title'] = df['Book_Title'].astype(str).str.lower().str.strip()
    df['Book_Author'] = df['Book_Author'].astype(str).str.lower().str.strip()
    df['Publisher'] = df['Publisher'].astype(str).str.lower().str.strip()
    df['ISBN'] = df['ISBN'].astype(str).str.strip()

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2025).astype(int)
    df['jumlah_pinjam'] = pd.to_numeric(df['jumlah_pinjam'], errors='coerce').fillna(0)

    df['book_age'] = 2025 - df['Year']
    df['title_length'] = df['Book_Title'].astype(str).str.len()
    df['title_word_count'] = df['Book_Title'].astype(str).str.split().str.len()
    df['isbn_length'] = df['ISBN'].str.replace('-', '', regex=False).str.len()
    df['isbn_prefix'] = df['ISBN'].str[:1].apply(lambda x: int(x) if str(x).isdigit() else -1)

    df['avg_word_length'] = df['Book_Title'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )

    df['decade'] = (df['book_age'] // 10) * 10

    # Encoding HARUS pakai LabelEncoder dari training (bukan cat.codes per-batch),
    # supaya kode author/publisher konsisten dengan yang dipelajari model.
    author_raw = df['Book_Author'].copy()
    publisher_raw = df['Publisher'].copy()

    if le_author is not None:
        df['Book_Author'] = author_raw.apply(lambda v: safe_encode(le_author, v))
    elif not pd.api.types.is_numeric_dtype(df['Book_Author']):
        df['Book_Author'] = df['Book_Author'].astype('category').cat.codes

    if le_publisher is not None:
        df['Publisher'] = publisher_raw.apply(lambda v: safe_encode(le_publisher, v))
    elif not pd.api.types.is_numeric_dtype(df['Publisher']):
        df['Publisher'] = df['Publisher'].astype('category').cat.codes

    # Frequency map di-keyed oleh kode HASIL ENCODING (train_model.py encode dulu,
    # baru hitung freq) — jadi mapping di sini juga harus pakai Book_Author yang
    # sudah di-encode, bukan nama mentah.
    if author_freq_map:
        df['author_freq'] = df['Book_Author'].map(author_freq_map).fillna(1)
    else:
        df['author_freq'] = df['Book_Author'].map(df['Book_Author'].value_counts()).fillna(1)

    if publisher_freq_map:
        df['publisher_freq'] = df['Publisher'].map(publisher_freq_map).fillna(1)
    else:
        df['publisher_freq'] = df['Publisher'].map(df['Publisher'].value_counts()).fillna(1)

    df['author_freq_log'] = np.log1p(df['author_freq'])
    df['publisher_freq_log'] = np.log1p(df['publisher_freq'])

    df['author_x_pub'] = df['Book_Author'] * df['Publisher']
    df['age_x_author'] = df['book_age'] * df['author_freq_log']

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Fitur tidak ditemukan: {missing}")

    return df[feature_names]


def transform_single_to_df(data, feature_names, le_author=None, le_publisher=None, author_map=None, publisher_map=None):
    df = pd.DataFrame([{
        'Book_Title': data['title'],
        'Book_Author': data['author'],
        'Publisher': data['publisher'],
        'Year': data['year'],
        'jumlah_pinjam': data['jumlah_pinjam'],
        'ISBN': "0"
    }])

    return build_model_input(df, feature_names, le_author, le_publisher, author_map, publisher_map)


def preprocess_batch(df, le_author=None, le_publisher=None, author_map=None, publisher_map=None):
    df = df.copy()
    df = df.fillna("unknown")
    return df


def evaluate_model(df, model, feature_names, le_author=None, le_publisher=None, author_map=None, publisher_map=None, mode='full'):
    df = df.copy()

    if 'kategori' not in df.columns:
        raise ValueError("Kolom 'kategori' tidak ditemukan")

    df = df.dropna(subset=['kategori'])

    if df.shape[0] == 0:
        raise ValueError("Dataset kosong setelah drop NA")

    if df['kategori'].nunique() < 2:
        raise ValueError("Label hanya 1 kelas")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df['kategori'])

    X = build_model_input(df, feature_names)

    if mode == 'split':
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model.fit(X_train, y_train)
    else:
        X_eval = X
        y_eval = y

    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_eval, y_pred)

    return {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, average='weighted', zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": le_target.classes_.tolist()
    }