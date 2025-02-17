#!/usr/bin/env bash
# Exit on error
set -o errexit

# Add Tesseract repository
apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:alex-p/tesseract-ocr5
apt-get update -y

# Install system dependencies
apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    libgl1-mesa-glx

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt 