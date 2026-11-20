#!/bin/sh
cd "$(dirname "$0")"
echo "Menyiapkan Library ML Dashboard (memuat model, tunggu sebentar)..."
.venv/bin/python app.py
