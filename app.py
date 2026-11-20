from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import datetime
import os
import json
import logging
import tempfile
import traceback
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ================================
# LOGGING SETUP (fallback ke temp jika OneDrive lock)
# ================================
def _log_path():
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')
    try:
        with open(local, 'a', encoding='utf-8') as f:
            f.write('')
        return local
    except OSError:
        return os.path.join(tempfile.gettempdir(), 'library_ml_app.log')

class SafeFileHandler(logging.FileHandler):
    def handleError(self, record):
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        SafeFileHandler(_log_path(), encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').disabled = True

# ================================
# IMPORT UTILS
# ================================
from utils import (
    build_single_input,
    transform_single_to_df,
    preprocess_batch,
    build_model_input,
    evaluate_model
)

# ================================
# CONSTANTS
# ================================
CURRENT_YEAR = datetime.datetime.now().year
MIN_YEAR = 1900
MAX_YEAR = CURRENT_YEAR

REQUIRED_BATCH_COLUMNS = ['Book_Title', 'Book_Author', 'Publisher', 'jumlah_pinjam', 'Year', 'Tanggal']

# ================================
# HELPER: JSON ERROR RESPONSE
# ================================
def error_response(message: str, status: int = 400, details=None) -> tuple:
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    logger.warning(f"[{status}] {message}" + (f" | details: {details}" if details else ""))
    return jsonify(payload), status

def success_response(data: dict) -> tuple:
    return jsonify({"success": True, **data}), 200

# ================================
# VALIDASI INPUT SINGLE PREDICT
# ================================
def validate_input(data: dict) -> list:
    errors = []

    try:
        jumlah_pinjam = float(data.get('jumlah_pinjam', ''))
        if jumlah_pinjam < 0:
            errors.append("jumlah_pinjam tidak boleh negatif")
        elif jumlah_pinjam > 1_000_000:
            errors.append("jumlah_pinjam melebihi batas wajar (> 1.000.000)")
    except (ValueError, TypeError):
        errors.append("jumlah_pinjam harus berupa angka valid")

    try:
        year = int(data.get('year', ''))
        if year < MIN_YEAR or year > MAX_YEAR:
            errors.append(f"year harus antara {MIN_YEAR} dan {MAX_YEAR}")
    except (ValueError, TypeError):
        errors.append("year harus berupa angka bulat valid")

    title = str(data.get('title', '')).strip()
    if not title:
        errors.append("title tidak boleh kosong")
    elif len(title) > 500:
        errors.append("title terlalu panjang (maksimal 500 karakter)")

    author = str(data.get('author', '')).strip()
    if author and len(author) > 300:
        errors.append("author terlalu panjang (maksimal 300 karakter)")

    publisher = str(data.get('publisher', '')).strip()
    if publisher and len(publisher) > 300:
        errors.append("publisher terlalu panjang (maksimal 300 karakter)")

    return errors


# ================================
# APP
# ================================
app = Flask(__name__)

# ================================
# LOAD MODEL
# ================================
def _find_model():
    candidates = [
        os.path.join(tempfile.gettempdir(), 'library_ml_model.pkl'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model.pkl'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

try:
    _model_path = _find_model()
    model_data = joblib.load(_model_path)
    model = model_data['model']
    feature_names = model_data.get('features') or model_data.get('feature_names')
    le_author = model_data.get('le_author')
    le_publisher = model_data.get('le_publisher')
    author_map = model_data.get('author_freq_map') or model_data.get('author_map', {})
    publisher_map = model_data.get('publisher_freq_map') or model_data.get('publisher_map', {})
    BEST_THRESHOLD = float(model_data.get('best_threshold', 0.5))
    logger.info(f"Model berhasil dimuat dari {_model_path} | threshold={BEST_THRESHOLD:.2f}")
except Exception as e:
    logger.critical(f"Gagal memuat model: {e}")
    raise SystemExit("Model tidak dapat dimuat. Pastikan models/model.pkl ada dan valid.")

def predict_label(X):
    """Prediksi 0/1 memakai threshold hasil tuning (sama dengan evaluasi di notebook), bukan 0.50 bawaan model.predict()."""
    return (model.predict_proba(X)[:, 1] >= BEST_THRESHOLD).astype(int)

# ================================
# LOAD DATASET
# ================================
def _find_dataset():
    candidates = [
        os.path.join(tempfile.gettempdir(), 'library_ml_dataset.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dataset.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]

DATA_PATH = _find_dataset()
logger.info(f"Dataset: {DATA_PATH}")
try:
    if os.path.exists(DATA_PATH):
        df_global = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset dimuat: {len(df_global)} baris dari {DATA_PATH}")
    else:
        df_global = pd.DataFrame()
        logger.warning(f"Dataset tidak ditemukan di {DATA_PATH}.")
except Exception as e:
    df_global = pd.DataFrame()
    logger.error(f"Gagal memuat dataset: {e}")

# ================================
# HITUNG EVAL RESULTS DARI MODEL
# ================================
def compute_eval_results(df: pd.DataFrame) -> dict:
    empty = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]], "labels": [0, 1]
    }
    results = {"full": empty.copy(), "split": empty.copy()}

    if df.empty or feature_names is None:
        logger.warning("compute_eval_results: dataset kosong atau feature_names tidak tersedia")
        return results

    try:
        target_col = None
        for candidate in ['kategori', 'label', 'target', 'popular', 'populer', 'is_popular']:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            logger.warning("compute_eval_results: kolom target tidak ditemukan di dataset")
            return results

        try:
            X = build_model_input(df, feature_names, le_author, le_publisher, author_map, publisher_map)
        except Exception as e:
            logger.error(f"build_model_input gagal: {e}")
            return results

        y = df[target_col].astype(int)

        if len(X) != len(y):
            logger.error("compute_eval_results: panjang X dan y tidak sama")
            return results

        best_threshold = model_data.get('best_threshold', 0.5)

        def apply_threshold(mdl, X_input):
            proba = mdl.predict_proba(X_input)[:, 1]
            return (proba >= best_threshold).astype(int)

        def build_metrics(y_true, y_pred) -> dict:
            cm = confusion_matrix(y_true, y_pred)
            cm_list = cm.tolist() if cm.shape == (2, 2) else [[0, 0], [0, 0]]
            return {
                "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
                "precision":        round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                "recall":           round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                "f1":               round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                "confusion_matrix": cm_list,
                "labels":           [0, 1]
            }

        y_pred_full = apply_threshold(model, X)
        results["full"] = build_metrics(y, y_pred_full)
        logger.info(f"Eval FULL — acc={results['full']['accuracy']:.4f} | f1={results['full']['f1']:.4f} | threshold={best_threshold}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred_split = apply_threshold(model, X_test)
        results["split"] = build_metrics(y_test, y_pred_split)
        logger.info(f"Eval SPLIT — acc={results['split']['accuracy']:.4f} | f1={results['split']['f1']:.4f}")

    except Exception as e:
        logger.error(f"compute_eval_results error: {e}\n{traceback.format_exc()}")

    return results


_saved_metrics = model_data.get('eval_metrics')
_empty_metrics = {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    "confusion_matrix": [[0, 0], [0, 0]], "labels": [0, 1]
}
if _saved_metrics:
    _m = {
        "accuracy":         _saved_metrics['accuracy'],
        "precision":        _saved_metrics['precision'],
        "recall":           _saved_metrics['recall'],
        "f1":               _saved_metrics['f1'],
        "confusion_matrix": _saved_metrics.get('confusion_matrix', [[0,0],[0,0]]),
        "labels":           [0, 1],
    }
    EVAL_RESULTS = {"full": _m, "split": _m}
    logger.info(f"EVAL_RESULTS dari model.pkl | acc={_saved_metrics['accuracy']}")
else:
    EVAL_RESULTS = {"full": _empty_metrics.copy(), "split": _empty_metrics.copy()}
    logger.warning("eval_metrics tidak ditemukan di model.pkl — jalankan ulang notebooks/train_model.ipynb")


# ================================
# HISTORY (FILE-BASED PERSISTENT)
# ================================
def _find_history_path():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'history.json')
    probe = base + '.probe'
    try:
        with open(probe, 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(probe)
        return base
    except OSError:
        pass
    fallback = os.path.join(tempfile.gettempdir(), 'library_ml_history.json')
    logger.warning(f"Project dir locked, history fallback: {fallback}")
    if os.path.exists(base) and not os.path.exists(fallback):
        try:
            import shutil
            shutil.copy2(base, fallback)
        except Exception:
            pass
    return fallback

HISTORY_FILE = _find_history_path()
logger.info(f"History file: {HISTORY_FILE}")

def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Gagal membaca {HISTORY_FILE}: {e}")
        return []

def save_history(data: list):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"History disimpan: {len(data)} entri ke {HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Gagal menyimpan {HISTORY_FILE}: {e}")


# ================================
# ROUTES — PAGES
# ================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/batch')
def batch_page():
    return render_template('batch_predict.html')

@app.route('/history-page')
def history_page():
    return render_template('history.html')


# ================================
# SINGLE PREDICT
# ================================
@app.route('/predict_auto', methods=['POST'])
def predict_auto():
    try:
        req = request.get_json(silent=True)
        if req is None:
            return error_response("Request body harus berupa JSON yang valid")

        errors = validate_input(req)
        if errors:
            return error_response("Validasi input gagal", status=400, details=errors)

        data = build_single_input(req)

        df_input = transform_single_to_df(
            data, feature_names, le_author, le_publisher, author_map, publisher_map
        )

        pred = predict_label(df_input)[0]
        label = "📚 Populer" if pred == 1 else "📉 Tidak Populer"

        logger.info(f"Prediksi sukses | title='{data['title']}' | result={label}")

        entry = {
            "time":          datetime.datetime.now().isoformat(),
            "title":         data['title'],
            "author":        data['author'],
            "publisher":     data['publisher'],
            "jumlah_pinjam": data['jumlah_pinjam'],
            "year":          data['year'],
            "result":        int(pred),
            "label":         label,
            "source":        "single"
        }

        _history = load_history()
        _history.append(entry)
        save_history(_history)
        logger.info(f"Single predict → history disimpan | total={len(_history)}")

        return success_response({"prediction": int(pred), "label": label})

    except Exception as e:
        logger.error(f"Predict error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan internal saat prediksi", status=500)


# ================================
# BATCH PREDICT
# ================================
@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        file = request.files.get('file')
        if not file:
            return error_response("File tidak ditemukan dalam request")

        if not file.filename.endswith('.csv'):
            return error_response("Hanya file .csv yang didukung")

        try:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            logger.warning(f"Gagal membaca CSV: {e}")
            return error_response(f"Gagal membaca file CSV: {str(e)}")

        missing = [col for col in REQUIRED_BATCH_COLUMNS if col not in df.columns]
        if missing:
            return error_response("Kolom tidak lengkap", details=missing)

        if df.empty:
            return error_response("File CSV kosong")

        logger.info(f"Batch file diterima: {len(df)} baris")

        # CLEAN DATA
        df['jumlah_pinjam'] = pd.to_numeric(df['jumlah_pinjam'], errors='coerce').fillna(0)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2000).astype(int)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

        df_result = df.copy()

        # ─── PREDIKSI PAKAI MODEL ───
        try:
            X_batch = build_model_input(df_result, feature_names, le_author, le_publisher, author_map, publisher_map)
            df_result['prediction'] = predict_label(X_batch)
            logger.info(f"Batch prediksi menggunakan threshold {BEST_THRESHOLD:.2f}")
        except Exception as e:
            logger.error(f"Prediksi batch gagal, fallback ke threshold: {e}")
            threshold = df_result['jumlah_pinjam'].median()
            df_result['prediction'] = (df_result['jumlah_pinjam'] >= threshold).astype(int)
            logger.warning("Batch menggunakan fallback threshold median")

        # ─── SIMPAN KE HISTORY (PERSISTENT) ───
        batch_time = datetime.datetime.now().isoformat()
        batch_entries = []
        original_filename = file.filename  # simpan sebelum stream ditutup

        for _, row in df_result.iterrows():
            pred = int(row.get('prediction', 0))
            entry = {
                "time":          batch_time,
                "title":         str(row.get('Book_Title', '')),
                "author":        str(row.get('Book_Author', '')),
                "publisher":     str(row.get('Publisher', '')),
                "jumlah_pinjam": int(row.get('jumlah_pinjam', 0)),
                "year":          int(row.get('Year', 0)),
                "result":        pred,
                "label":         "📚 Populer" if pred == 1 else "📉 Tidak Populer",
                "source":        "batch",
                "filename":      original_filename
            }
            batch_entries.append(entry)

        _history = load_history()
        _history.extend(batch_entries)
        save_history(_history)
        logger.info(f"Batch → history disimpan | +{len(batch_entries)} entri | total={len(_history)}")

        # ─── STATISTIK ───
        pop   = int((df_result['prediction'] == 1).sum())
        tidak = int((df_result['prediction'] == 0).sum())

        dist       = df['jumlah_pinjam'].value_counts().sort_index()
        df_tanggal = df_result.dropna(subset=['Tanggal'])
        week       = df_tanggal.groupby(df_tanggal['Tanggal'].dt.isocalendar().week)['jumlah_pinjam'].sum()
        month      = df_tanggal.groupby(df_tanggal['Tanggal'].dt.month)['jumlah_pinjam'].sum()
        year_group = df_tanggal.groupby(df_tanggal['Tanggal'].dt.year)['jumlah_pinjam'].sum()

        top_books     = df.groupby('Book_Title')['jumlah_pinjam'].sum().sort_values(ascending=False).head(5)
        top_author    = df.groupby('Book_Author')['jumlah_pinjam'].sum().sort_values(ascending=False).head(5)
        top_publisher = df.groupby('Publisher')['jumlah_pinjam'].sum().sort_values(ascending=False).head(5)

        df_result['Tanggal'] = df_result['Tanggal'].astype(str)
        df_result = df_result.replace({np.nan: None, np.inf: None, -np.inf: None})

        logger.info(f"Batch selesai | total={len(df_result)} | populer={pop} | tidak={tidak}")

        return success_response({
            "populer":              pop,
            "tidak":                tidak,
            "rows":                 df_result.to_dict(orient="records"),
            "dist_labels":          dist.index.astype(str).tolist(),
            "dist_values":          [int(v) for v in dist.values],
            "labels_week":          week.index.astype(str).tolist(),
            "values_week":          [int(v) for v in week.values],
            "labels_month":         month.index.astype(str).tolist(),
            "values_month":         [int(v) for v in month.values],
            "labels_year":          year_group.index.astype(str).tolist(),
            "values_year":          [int(v) for v in year_group.values],
            "top_books_labels":     top_books.index.tolist(),
            "top_books_values":     [int(v) for v in top_books.values],
            "top_author_labels":    top_author.index.tolist(),
            "top_author_values":    [int(v) for v in top_author.values],
            "top_publisher_labels": top_publisher.index.tolist(),
            "top_publisher_values": [int(v) for v in top_publisher.values]
        })

    except Exception as e:
        logger.error(f"Batch error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan internal saat batch predict", status=500)


# ================================
# MODEL EVALUATION
# ================================
@app.route('/model-evaluation')
def model_eval():
    try:
        mode = request.args.get('mode', 'split')

        if mode not in EVAL_RESULTS:
            return error_response(f"Mode tidak valid. Pilihan: {list(EVAL_RESULTS.keys())}")

        result = EVAL_RESULTS[mode]
        safe = {
            "accuracy":         float(result["accuracy"]),
            "precision":        float(result["precision"]),
            "recall":           float(result["recall"]),
            "f1":               float(result["f1"]),
            "confusion_matrix": [[int(cell) for cell in row] for row in result["confusion_matrix"]],
            "labels":           result["labels"]
        }

        logger.info(f"Model evaluation diminta | mode={mode} | acc={safe['accuracy']:.4f}")
        return jsonify(safe), 200

    except Exception as e:
        logger.error(f"Model eval error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan saat mengambil evaluasi model", status=500)


# ================================
# HISTORY
# ================================
@app.route('/history', methods=['GET'])
def get_history():
    _history = load_history()
    logger.info(f"History diminta | total={len(_history)} entri")
    return jsonify(_history[::-1]), 200

@app.route('/history', methods=['DELETE'])
def clear_history():
    save_history([])
    logger.info("History dihapus semua")
    return jsonify({"success": True, "message": "History berhasil dihapus"}), 200


# ================================
# DASHBOARD DATA
# ================================
@app.route('/dashboard-data')
def dashboard_data():
    try:
        empty_response = {
            "kategori": [0, 0],
            "tahun_labels": [], "tahun_values": [],
            "author_labels": [], "author_values": [],
            "publisher_labels": [], "publisher_values": []
        }

        if df_global.empty:
            logger.warning("Dashboard: dataset kosong")
            return jsonify(empty_response), 200

        df = df_global.copy()

        if 'jumlah_pinjam' not in df.columns:
            df['jumlah_pinjam'] = 0

        target_col = None
        for candidate in ['kategori', 'label', 'target', 'popular', 'populer', 'is_popular']:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col:
            kategori = df[target_col].value_counts()
            kategori_data = [int(kategori.get(0, 0)), int(kategori.get(1, 0))]
        else:
            kategori_data = [0, 0]

        tahun = (
            df.groupby('Year')['jumlah_pinjam'].sum().sort_index()
            if 'Year' in df.columns else pd.Series(dtype=float)
        )
        top_author = (
            df.groupby('Book_Author')['jumlah_pinjam'].sum().sort_values(ascending=False).head(5)
            if 'Book_Author' in df.columns else pd.Series(dtype=float)
        )
        top_publisher = (
            df.groupby('Publisher')['jumlah_pinjam'].sum().sort_values(ascending=False).head(5)
            if 'Publisher' in df.columns else pd.Series(dtype=float)
        )

        logger.info("Dashboard data berhasil dikirim")

        return jsonify({
            "kategori":         kategori_data,
            "tahun_labels":     tahun.index.astype(str).tolist(),
            "tahun_values":     [int(v) for v in tahun.values],
            "author_labels":    top_author.index.astype(str).tolist(),
            "author_values":    [int(v) for v in top_author.values],
            "publisher_labels": top_publisher.index.astype(str).tolist(),
            "publisher_values": [int(v) for v in top_publisher.values]
        }), 200

    except Exception as e:
        logger.error(f"Dashboard error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan saat memuat dashboard", status=500)


# ================================
# FEATURE IMPORTANCE
# ================================
@app.route('/feature-importance')
def feature_importance():
    try:
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model tidak memiliki feature_importances_")
            return jsonify([]), 200

        importances = model.feature_importances_
        result = sorted(
            [
                {"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
                for i in range(len(feature_names))
            ],
            key=lambda x: x["importance"],
            reverse=True
        )

        logger.info(f"Feature importance diminta | {len(result)} fitur")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Feature importance error: {e}\n{traceback.format_exc()}")
        return error_response("Gagal mengambil feature importance", status=500)


# ================================
# AUTOCOMPLETE
# ================================
@app.route('/autocomplete')
def autocomplete():
    try:
        query = request.args.get('q', '').strip().lower()
        field = request.args.get('field', 'title')

        if not query:
            return jsonify([]), 200

        if df_global.empty:
            return jsonify([]), 200

        field_map = {
            'title': 'Book_Title',
            'author': 'Book_Author',
            'publisher': 'Publisher'
        }

        col = field_map.get(field)
        if col is None:
            return error_response(f"field tidak valid. Pilihan: {list(field_map.keys())}")

        if col not in df_global.columns:
            return jsonify([]), 200

        results = (
            df_global[col].astype(str)
            [df_global[col].astype(str).str.lower().str.contains(query, na=False)]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Autocomplete error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan pada autocomplete", status=500)


# ================================
# BOOK DETAIL
# ================================
@app.route('/book-detail')
def book_detail():
    try:
        title = request.args.get('title', '').strip().lower()

        if not title:
            return error_response("Parameter 'title' tidak boleh kosong")

        if df_global.empty:
            return jsonify({}), 200

        df = df_global.copy()
        mask = df['Book_Title'].str.lower().str.contains(title, na=False)
        row = df[mask]

        if row.empty:
            logger.info(f"Book detail: tidak ditemukan untuk query='{title}'")
            return jsonify({}), 200

        row = row.iloc[0]

        return jsonify({
            "title":         str(row.get('Book_Title', '')),
            "author":        str(row.get('Book_Author', '')),
            "publisher":     str(row.get('Publisher', '')),
            "year":          int(row.get('Year', 2000)),
            "jumlah_pinjam": int(row.get('jumlah_pinjam', 0))
        }), 200

    except Exception as e:
        logger.error(f"Book detail error: {e}\n{traceback.format_exc()}")
        return error_response("Terjadi kesalahan saat mengambil detail buku", status=500)


# ================================
# RUN
# ================================
if __name__ == '__main__':
    # Port 5000 dipakai AirPlay Receiver di macOS; bisa diganti lewat env PORT
    port = int(os.environ.get('PORT', 5001))

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            print(f"\n  Port {port} sedang dipakai — kemungkinan dashboard sudah berjalan di jendela Terminal lain.")
            print(f"  Buka http://localhost:{port} atau tutup dulu yang lama (Ctrl + C), lalu jalankan lagi.")
            print(f"  Bisa juga pakai port lain:  PORT=5002 ./run.sh\n")
            raise SystemExit(1)

    print("\n" + "=" * 52)
    print("  Library ML Dashboard sudah berjalan!")
    print(f"  Buka di browser : http://localhost:{port}")
    print("  Untuk berhenti  : tekan Ctrl + C")
    print("=" * 52 + "\n", flush=True)
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)