import os
import json
import re
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import csv
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'pdf'}

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Using blank English model")
    nlp = spacy.blank("en")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    return text

def preprocess_with_spacy(text):
    """Preprocess text using spaCy: lemmatize, remove stopwords & punctuation."""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
    ]
    return " ".join(tokens)

def extract_skills(text):
    """Extract skills and key terms from text."""
    tech_skills = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'django', 'flask', 'fastapi', 'sql', 'mysql', 'postgresql', 'mongodb',
        'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'linux',
        'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch', 'scikit',
        'pandas', 'numpy', 'spark', 'kafka', 'rest', 'api', 'microservices',
        'agile', 'scrum', 'ci/cd', 'devops', 'html', 'css', 'php', 'ruby', 'go',
        'rust', 'c++', 'c#', 'swift', 'kotlin', 'excel', 'tableau', 'power bi',
        'data analysis', 'data science', 'blockchain', 'cybersecurity', 'networking'
    ]
    text_lower = text.lower()
    found = [skill for skill in tech_skills if skill in text_lower]
    return found

def extract_experience_years(text):
    """Extract years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
    ]
    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            years = int(match)
            if years > max_years and years < 50:
                max_years = years
    return max_years

def extract_education(text):
    """Detect education level from text."""
    text_lower = text.lower()
    if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
        return 'PhD', 4
    elif any(term in text_lower for term in ['master', 'msc', 'm.sc', 'mba', 'm.tech', 'me ']):
        return 'Masters', 3
    elif any(term in text_lower for term in ['bachelor', 'bsc', 'b.sc', 'be ', 'b.tech', 'b.e.']):
        return 'Bachelors', 2
    elif any(term in text_lower for term in ['diploma', 'associate']):
        return 'Diploma', 1
    return 'Not specified', 0

def score_resume(resume_text, job_description, processed_resume, processed_jd):
    """Multi-dimensional scoring algorithm."""
    scores = {}

    # 1. TF-IDF Cosine Similarity (40%)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_jd, processed_resume])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        scores['tfidf_score'] = round(float(cosine_sim) * 100, 2)
    except:
        scores['tfidf_score'] = 0

    # 2. Keyword Match Score (30%)
    jd_words = set(processed_jd.split())
    resume_words = set(processed_resume.split())
    if len(jd_words) > 0:
        keyword_match = len(jd_words.intersection(resume_words)) / len(jd_words)
        scores['keyword_score'] = round(keyword_match * 100, 2)
    else:
        scores['keyword_score'] = 0

    # 3. Skills Match (20%)
    jd_skills = set(extract_skills(job_description))
    resume_skills = set(extract_skills(resume_text))
    matched_skills = jd_skills.intersection(resume_skills)
    if len(jd_skills) > 0:
        skill_score = len(matched_skills) / len(jd_skills)
        scores['skill_score'] = round(skill_score * 100, 2)
    else:
        scores['skill_score'] = 50  # neutral if no skills found in JD

    # 4. Experience Score (10%)
    exp_years = extract_experience_years(resume_text)
    exp_score = min(exp_years * 10, 100)  # 10 points per year, capped at 100
    scores['experience_score'] = exp_score
    scores['experience_years'] = exp_years

    # Education info (bonus metadata)
    edu_level, edu_points = extract_education(resume_text)
    scores['education'] = edu_level

    # Weighted Final Score
    final = (
        scores['tfidf_score'] * 0.40 +
        scores['keyword_score'] * 0.30 +
        scores['skill_score'] * 0.20 +
        scores['experience_score'] * 0.10
    )
    scores['final_score'] = round(final, 2)
    scores['matched_skills'] = list(matched_skills)
    scores['all_skills'] = list(resume_skills)

    return scores

def get_rank_label(score):
    if score >= 75:
        return 'Excellent', '#22c55e'
    elif score >= 55:
        return 'Good', '#3b82f6'
    elif score >= 35:
        return 'Average', '#f59e0b'
    else:
        return 'Below Average', '#ef4444'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rank', methods=['POST'])
def rank_resumes():
    if 'resumes' not in request.files:
        return jsonify({'error': 'No resume files uploaded'}), 400

    job_description = request.form.get('job_description', '').strip()
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400

    files = request.files.getlist('resumes')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'Please upload at least one resume'}), 400

    # Preprocess job description
    processed_jd = preprocess_with_spacy(job_description)
    jd_skills = extract_skills(job_description)

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract and process resume
            resume_text = extract_text_from_pdf(filepath)
            processed_resume = preprocess_with_spacy(resume_text)

            # Score it
            scores = score_resume(resume_text, job_description, processed_resume, processed_jd)
            label, color = get_rank_label(scores['final_score'])

            results.append({
                'filename': filename,
                'final_score': scores['final_score'],
                'tfidf_score': scores['tfidf_score'],
                'keyword_score': scores['keyword_score'],
                'skill_score': scores['skill_score'],
                'experience_score': scores['experience_score'],
                'experience_years': scores['experience_years'],
                'education': scores['education'],
                'matched_skills': scores['matched_skills'],
                'all_skills': scores['all_skills'][:15],
                'rank_label': label,
                'rank_color': color
            })

    # Sort by final score descending
    results.sort(key=lambda x: x['final_score'], reverse=True)

    # Add rank numbers
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return jsonify({
        'results': results,
        'jd_skills': jd_skills,
        'total': len(results),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    results = data.get('results', [])
    job_title = data.get('job_title', 'Position')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['AI-Powered Resume Ranking Report'])
    writer.writerow([f'Job Profile: {job_title}'])
    writer.writerow([f'Generated: {timestamp}'])
    writer.writerow([f'Total Candidates: {len(results)}'])
    writer.writerow([])

    # Column headers
    writer.writerow([
        'Rank', 'Resume File', 'Final Score (%)', 'Rating',
        'TF-IDF Score', 'Keyword Score', 'Skill Score', 'Experience Score',
        'Experience (Years)', 'Education', 'Matched Skills'
    ])

    for r in results:
        writer.writerow([
            r.get('rank', ''),
            r.get('filename', ''),
            r.get('final_score', ''),
            r.get('rank_label', ''),
            r.get('tfidf_score', ''),
            r.get('keyword_score', ''),
            r.get('skill_score', ''),
            r.get('experience_score', ''),
            r.get('experience_years', ''),
            r.get('education', ''),
            ', '.join(r.get('matched_skills', []))
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'resume_ranking_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
