#!/usr/bin/env bash
set -e

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_DIR="$(pwd)/models"
mkdir -p "$MODEL_DIR"

# Map Indic codes to tesseract-ocr/tessdata_best URLs
declare -A BEST_MODELS=(
  [ben]="https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata"
  [guj]="https://github.com/tesseract-ocr/tessdata_best/raw/main/guj.traineddata"
  [hin]="https://github.com/tesseract-ocr/tessdata_best/raw/main/hin.traineddata"
  [kan]="https://github.com/tesseract-ocr/tessdata_best/raw/main/kan.traineddata"
  [mal]="https://github.com/tesseract-ocr/tessdata_best/raw/main/mal.traineddata"
  [mni]="https://github.com/tesseract-ocr/tessdata_best/raw/main/mni.traineddata"
  [ori]="https://github.com/tesseract-ocr/tessdata_best/raw/main/ori.traineddata"
  [pan]="https://github.com/tesseract-ocr/tessdata_best/raw/main/pan.traineddata"
  [sat]="https://github.com/tesseract-ocr/tessdata_best/raw/main/sat.traineddata"
  [tam]="https://github.com/tesseract-ocr/tessdata_best/raw/main/tam.traineddata"
  [tel]="https://github.com/tesseract-ocr/tessdata_best/raw/main/tel.traineddata"
)

# -------------------------------------------------------------------
# 0) Clean up any existing Tesseract installs & local models
# -------------------------------------------------------------------
echo "0) Purging old Tesseract packages..."
sudo apt-get purge --auto-remove -y "tesseract-ocr*" "libtesseract*" "libleptonica*" || true
sudo rm -rf /usr/share/tesseract-ocr /usr/local/share/tessdata
rm -rf "$MODEL_DIR"

# Recreate model directory
mkdir -p "$MODEL_DIR"

# -------------------------------------------------------------------
# 1) Install Tesseract 5.x
# -------------------------------------------------------------------
echo "1) Installing Tesseract 5.x..."
sudo apt-get update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list*; then
  sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev

echo -n "   Tesseract version: "
tesseract --version | head -n1

# -------------------------------------------------------------------
# 2) Install Ubuntu language packs
# -------------------------------------------------------------------
echo "2) Installing language packs for: ${TESSERACT_LANG//+/ }"
if [[ "${IS_MULTILINGUAL,,}" == "true" ]]; then
  for lang in ${TESSERACT_LANG//+/ }; do
    if [[ "$lang" == "eng" ]]; then
      echo "   • eng: default"
      continue
    fi
    pkg="tesseract-ocr-$lang"
    echo -n "   • $pkg: "
    if sudo apt-get install -y "$pkg" &>/dev/null; then
      echo "installed"
    else
      echo "not available"
    fi
  done
else
  echo "   Multilingual disabled; skipping packs."
fi

# -------------------------------------------------------------------
# 3) Download & install “best” tessdata models for Indic
# -------------------------------------------------------------------
echo "3) Fetching ‘best’ tessdata into $MODEL_DIR"
for code in ben guj hin kan mal mni ori pan sat tam tel; do
  if [[ " ${TESSERACT_LANG//+/ } " =~ " $code " ]]; then
    url="${BEST_MODELS[$code]}"
    outfile="$MODEL_DIR/$code.traineddata"
    echo -n "   • $code: "
    if wget -qO "$outfile" "$url"; then
      if [[ -s "$outfile" ]]; then
        echo "downloaded"
      else
        echo "empty; removing"
        rm -f "$outfile"
      fi
    else
      echo "download failed"
      rm -f "$outfile"
    fi
  fi
done

# -------------------------------------------------------------------
# 4) Final Verification & Size Listing
# -------------------------------------------------------------------
echo
echo "4) System Tesseract languages:"
tesseract --list-langs || echo "   (could not list)"

echo
echo "5) ‘Best’ models in $MODEL_DIR:"
found=false
for f in "$MODEL_DIR"/*.traineddata; do
  [[ -e "$f" ]] || continue
  size_kb=$(( $(stat -c "%s" "$f") / 1024 ))
  echo "   • $(basename "$f"): ${size_kb} KB"
  found=true
done
$found || echo "   (no models downloaded)"

echo
echo "Setup complete. Use these with:"
grep -qxF 'export TESSERACT_CMD=$(which tesseract)' ~/.bashrc || echo 'export TESSERACT_CMD=$(which tesseract)' >> ~/.bashrc
echo "   tesseract image.png out -l ${TESSERACT_LANG} --tessdata-dir $MODEL_DIR"



