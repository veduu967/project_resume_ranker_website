"""
scoring.py — Standalone Resume Scoring Algorithm
Can be imported or run directly:
    python scoring.py

This module encapsulates the full NLP pipeline:
  1. PDF Text Extraction (PyPDF2)
  2. Preprocessing (SpaCy lemmatization + stopword removal)
  3. TF-IDF Vectorization (scikit-learn)
  4. Cosine Similarity scoring
  5. Keyword, Skill, and Experience scoring
  6. Weighted final score & ranking
"""

import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# ──────────────────────────────────────────────
# Scoring Weights (must sum to 1.0)
# ──────────────────────────────────────────────
WEIGHTS = {
    'tfidf':      0.40,   # Semantic similarity via TF-IDF
    'keyword':    0.30,   # Direct keyword overlap
    'skill':      0.20,   # Technical skills match
    'experience': 0.10,   # Years of experience
}

TECH_SKILL_LIBRARY = [
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
    'node', 'django', 'flask', 'fastapi', 'sql', 'mysql', 'postgresql', 'mongodb',
    'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'linux',
    'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch', 'scikit',
    'pandas', 'numpy', 'spark', 'kafka', 'rest', 'api', 'microservices',
    'agile', 'scrum', 'ci/cd', 'devops', 'html', 'css', 'php', 'ruby', 'go',
    'rust', 'c++', 'c#', 'swift', 'kotlin', 'tableau', 'power bi',
    'data analysis', 'data science', 'blockchain', 'cybersecurity',
]

# Load spaCy once at module level
_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ──────────────────────────────────────────────
# Step 1: Text Extraction
# ──────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


# ──────────────────────────────────────────────
# Step 2: SpaCy Preprocessing
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Lowercase → tokenize → lemmatize → remove stopwords & punctuation.
    Returns a clean space-joined string ready for TF-IDF.
    """
    nlp = _get_nlp()
    doc = nlp(text.lower())
    tokens = [
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 2
    ]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# Step 3: Feature Extractors
# ──────────────────────────────────────────────
def extract_skills(text: str) -> list[str]:
    """Match skills from TECH_SKILL_LIBRARY against raw text."""
    lower = text.lower()
    return [skill for skill in TECH_SKILL_LIBRARY if skill in lower]


def extract_experience_years(text: str) -> int:
    """Pull the max years-of-experience number from text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
    ]
    max_yrs = 0
    for pattern in patterns:
        for m in re.findall(pattern, text.lower()):
            y = int(m)
            if y < 50:
                max_yrs = max(max_yrs, y)
    return max_yrs


def extract_education(text: str) -> tuple[str, int]:
    """Return (education label, numeric level 0–4)."""
    lower = text.lower()
    if any(t in lower for t in ['phd', 'ph.d', 'doctorate', 'doctoral']):
        return 'PhD', 4
    if any(t in lower for t in ['master', 'msc', 'm.sc', 'mba', 'm.tech']):
        return 'Masters', 3
    if any(t in lower for t in ['bachelor', 'bsc', 'b.sc', 'b.tech', 'b.e.']):
        return 'Bachelors', 2
    if any(t in lower for t in ['diploma', 'associate']):
        return 'Diploma', 1
    return 'Not specified', 0


# ──────────────────────────────────────────────
# Step 4: Scoring Algorithm
# ──────────────────────────────────────────────
def score_resume(resume_text: str, job_description: str) -> dict:
    """
    Score a resume against a job description across 4 dimensions.

    Returns a dict with all sub-scores and a weighted final_score (0–100).
    """
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_description)

    # --- TF-IDF Cosine Similarity (40%) ---
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    try:
        matrix = vectorizer.fit_transform([processed_jd, processed_resume])
        tfidf_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100
    except Exception:
        tfidf_score = 0.0

    # --- Keyword Overlap (30%) ---
    jd_words = set(processed_jd.split())
    res_words = set(processed_resume.split())
    keyword_score = (len(jd_words & res_words) / len(jd_words) * 100) if jd_words else 0.0

    # --- Skill Match (20%) ---
    jd_skills  = set(extract_skills(job_description))
    res_skills = set(extract_skills(resume_text))
    matched    = jd_skills & res_skills
    skill_score = (len(matched) / len(jd_skills) * 100) if jd_skills else 50.0

    # --- Experience Score (10%) ---
    years = extract_experience_years(resume_text)
    experience_score = min(years * 10, 100)

    # --- Weighted Final ---
    final = (
        tfidf_score      * WEIGHTS['tfidf'] +
        keyword_score    * WEIGHTS['keyword'] +
        skill_score      * WEIGHTS['skill'] +
        experience_score * WEIGHTS['experience']
    )

    edu_label, _ = extract_education(resume_text)

    return {
        'final_score':       round(final, 2),
        'tfidf_score':       round(tfidf_score, 2),
        'keyword_score':     round(keyword_score, 2),
        'skill_score':       round(skill_score, 2),
        'experience_score':  round(experience_score, 2),
        'experience_years':  years,
        'education':         edu_label,
        'matched_skills':    sorted(matched),
        'resume_skills':     sorted(res_skills),
        'jd_skills':         sorted(jd_skills),
    }


# ──────────────────────────────────────────────
# Step 5: Batch Ranking
# ──────────────────────────────────────────────
def rank_resumes(pdf_paths: list[str], job_description: str) -> list[dict]:
    """
    Score and rank a list of PDF resumes against a job description.
    Returns a list of result dicts sorted best → worst.
    """
    results = []
    for path in pdf_paths:
        try:
            text = extract_pdf_text(path)
            scores = score_resume(text, job_description)
            scores['file'] = path
            results.append(scores)
        except Exception as e:
            results.append({'file': path, 'final_score': 0, 'error': str(e)})

    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1
    return results


# ──────────────────────────────────────────────
# CLI Demo
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import sys, json, glob

    JOB_DESCRIPTION = """
    We are looking for a Senior Python Backend Engineer with 5+ years of experience.
    Must have strong skills in Flask, REST API design, PostgreSQL, Docker, and AWS.
    Experience with machine learning pipelines, scikit-learn, and CI/CD is highly preferred.
    Familiarity with Agile/Scrum methodologies required.
    """

    pdfs = glob.glob('sample_resumes/*.pdf')
    if not pdfs:
        print("No PDFs found in sample_resumes/. Run generate_samples.py first.")
        sys.exit(1)

    print(f"\nRanking {len(pdfs)} resumes...\n{'='*60}")
    results = rank_resumes(pdfs, JOB_DESCRIPTION)

    for r in results:
        print(f"\nRank #{r['rank']}: {r['file']}")
        print(f"  Final Score   : {r.get('final_score', 'N/A')}%")
        print(f"  TF-IDF        : {r.get('tfidf_score', 'N/A')}%")
        print(f"  Keyword Match : {r.get('keyword_score', 'N/A')}%")
        print(f"  Skill Match   : {r.get('skill_score', 'N/A')}%")
        print(f"  Experience    : {r.get('experience_years', 0)} yrs")
        print(f"  Education     : {r.get('education', 'N/A')}")
        print(f"  Matched Skills: {', '.join(r.get('matched_skills', []))}")
    print(f"\n{'='*60}")
