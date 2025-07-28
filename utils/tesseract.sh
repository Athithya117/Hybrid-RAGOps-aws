

#!/usr/bin/env bash
set -e

# default tesseract 4.1.1 in ubuntu-22.04 is very old(2019), so removing
# sudo apt purge --autoremove -y tesseract-ocr tesseract-ocr-* libtesseract* libleptonica*


# --- Add PPA for latest Tesseract 5.x ---
sudo apt update -y
if ! grep -q "^deb .\+ppa.launchpadcontent.net/alex-p/tesseract-ocr5" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
  sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
fi

sudo apt update -y
sudo apt install -y tesseract-ocr

# --- Install additional languages if needed ---
if [[ "$IS_MULTILINGUAL" == "true" ]]; then
  langs=$(echo "$TESSERACT_LANG" | tr '+' ' ')
  for code in $langs; do
    if [[ "$code" == "eng" ]]; then
      continue  # English is installed by default
    fi
    pkg="tesseract-ocr-${code}"
    echo "Installing language pack: $pkg"
    if sudo apt install -y "$pkg"; then
      echo "Installed: $pkg"
    else
      echo "Warning: package '$pkg' not found or failed to install"
    fi
  done
else
  echo "IS_MULTILINGUAL=false — skipping language pack installs"
fi

# --- Confirm installation ---
echo ""
echo "Tesseract version:"
tesseract --version

echo ""
echo "Available languages:"
tesseract --list-langs || echo "Failed to list languages"

# --- Optional: Remove unwanted log lines (cleanup for Python logs or similar) ---
find . -type f -exec sed -i '/log\.info(f"Uploaded page .*→ s3:\/\/.*")/c\\' {} +
