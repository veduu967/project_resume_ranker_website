# 🧠 ResumeIQ — AI-Powered Resume Ranker

Rank resumes intelligently against any job description using NLP: SpaCy, TF-IDF, and a multi-dimensional scoring algorithm.

---

## 📁 Project Structure

```
resume_ranker/
├── app.py               ← Flask web application (main entry point)
├── scoring.py           ← Standalone NLP scoring module (importable)
├── generate_samples.py  ← Generates sample PDF resumes for testing
├── requirements.txt     ← Python dependencies
├── setup.sh             ← One-command setup & launch
├── templates/
│   └── index.html       ← UI (drag-drop upload, live results, HR export)
├── uploads/             ← Temporary PDF storage (auto-created)
└── sample_resumes/      ← Generated test resumes (after running generate_samples.py)
```

---

## ⚡ Quick Start

```bash
# One-command setup (creates venv, installs deps, generates samples, starts app)
bash setup.sh

# Or manually:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python generate_samples.py   # create test PDFs
python app.py                # start Flask on http://localhost:5000
```

---

## 🔬 How the Scoring Algorithm Works

Each resume is scored across **4 weighted dimensions**:

| Dimension | Weight | Method |
|---|---|---|
| **TF-IDF Cosine Similarity** | 40% | Vectorize JD + resume, compute cosine similarity |
| **Keyword Match** | 30% | Lemmatized token overlap between JD and resume |
| **Skill Match** | 20% | Match against 40+ tech skill library |
| **Experience** | 10% | Regex extraction of years of experience |

```
Final Score = (TF-IDF × 0.40) + (Keyword × 0.30) + (Skill × 0.20) + (Exp × 0.10)
```

### SpaCy Preprocessing Pipeline
```
Raw Text → Lowercase → Tokenize → Lemmatize → Remove Stopwords & Punctuation → Clean Tokens
```

### Rating Thresholds
| Score | Label |
|---|---|
| ≥ 75 | 🟢 Excellent |
| 55–74 | 🔵 Good |
| 35–54 | 🟡 Average |
| < 35 | 🔴 Below Average |

---

## 🌐 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/rank` | Upload resumes + JD → JSON rankings |
| `POST` | `/download_report` | Generate downloadable CSV HR report |

### POST /rank
```
Content-Type: multipart/form-data
Fields:
  - job_description: string (required)
  - resumes: file[] (PDF, required)

Response: {
  results: [...],  // sorted by score
  jd_skills: [],
  total: int,
  timestamp: string
}
```

---

## 🐍 Using scoring.py Standalone

```python
from scoring import rank_resumes, score_resume, extract_pdf_text

# Score a single resume
text = extract_pdf_text("resume.pdf")
scores = score_resume(text, job_description="We need a Python Flask engineer...")
print(scores['final_score'])  # e.g. 72.4

# Rank multiple resumes
results = rank_resumes(
    pdf_paths=["alice.pdf", "bob.pdf", "carol.pdf"],
    job_description="Senior Python Backend Engineer with Flask, AWS, Docker..."
)
for r in results:
    print(f"#{r['rank']} {r['file']} → {r['final_score']}%")
```

---

## 📊 Sample Output

```
Rank #1: alice_chen_senior_python.pdf    → 81.2%  (Excellent)
Rank #2: david_kumar_fullstack.pdf       → 68.5%  (Good)
Rank #3: bob_sharma_mid_backend.pdf      → 54.3%  (Good)
Rank #4: carol_jones_junior_dev.pdf      → 28.1%  (Below Average)
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Web Framework | Flask |
| NLP Processing | SpaCy (en_core_web_sm) |
| Vectorization | scikit-learn TF-IDF |
| Similarity | Cosine Similarity |
| PDF Extraction | PyPDF2 |
| HR Export | Python csv module |
| Frontend | Vanilla JS + CSS (dark UI) |

---

## 📈 Extending the Project

- **Add OCR** for scanned PDFs: `pytesseract` + `pdf2image`
- **Sentence Transformers** for semantic similarity (BERT-based)
- **Named Entity Recognition** to extract candidate name/email automatically
- **Database storage**: store rankings in SQLite/PostgreSQL
- **Authentication**: add HR login with Flask-Login
- **More export formats**: PDF reports using reportlab
