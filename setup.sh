#!/bin/bash
# setup.sh — One-command setup for ResumeIQ
# Usage: bash setup.sh

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     ResumeIQ — AI Resume Ranker      ║"
echo "║         Setup & Launch Script        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Create virtual environment
echo "→ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
echo "→ Installing Python packages..."
pip install --upgrade pip -q
pip install flask PyPDF2 spacy scikit-learn numpy werkzeug reportlab -q

# 3. Download spaCy model
echo "→ Downloading SpaCy language model (en_core_web_sm)..."
python -m spacy download en_core_web_sm -q

# 4. Create uploads directory
mkdir -p uploads sample_resumes

# 5. Generate sample resumes
echo "→ Generating sample PDF resumes..."
python generate_samples.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  Starting Flask app on port 5000...  ║"
echo "║  Open: http://localhost:5000         ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 6. Launch Flask
python app.py
