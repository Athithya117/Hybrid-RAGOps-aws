#!/usr/bin/env bash

set -e

# --- Core Tesseract install ---
sudo apt update -y
sudo apt install -y tesseract-ocr

# --- Conditional language installs ---
if [[ "$IS_MULTILINGUAL" == "true" ]]; then

  langs=$(echo "$TESSERACT_LANG" | tr '+' ' ')
  for code in $langs; do
    if [[ "$code" == "eng" ]]; then
      continue  # already included by default tesseract-ocr
    fi
    pkg="tesseract-ocr-${code}"
    echo "Installing language pack: $pkg"
    if sudo apt install -y "$pkg"; then
      echo "Installed: $pkg"
    else
      echo "Warning: package '$pkg' not found or failed install"
    fi
  done
else
  echo "IS_MULTILINGUAL=false — skipping language pack installs"
fi

# --- Confirm installed languages ---
echo ""
echo "Tesseract available languages:"
tesseract --list-langs || echo "tesseract not found or failed"
