"""
Generate sample PDF resumes for testing the Resume Ranker.
Run: python generate_samples.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

resumes = [
    {
        "filename": "alice_chen_senior_python.pdf",
        "content": """
ALICE CHEN
alice.chen@email.com | linkedin.com/in/alicechen | GitHub: alicechen

PROFESSIONAL SUMMARY
Senior Python Backend Engineer with 7 years of experience building scalable REST APIs, microservices, and ML pipelines. Expert in Flask, Django, FastAPI, PostgreSQL, Docker, and AWS. Passionate about clean code and data-driven engineering.

EXPERIENCE

Senior Software Engineer — DataTech Inc. (2020–Present)
- Built REST APIs using Flask and FastAPI serving 2M+ requests/day
- Designed machine learning pipelines using scikit-learn, pandas, and numpy
- Deployed microservices on AWS ECS with Docker and Kubernetes
- Implemented CI/CD pipelines using GitHub Actions and Jenkins
- Optimized PostgreSQL queries reducing latency by 60%

Software Engineer — WebSolutions Ltd. (2017–2020)
- Developed Django applications with PostgreSQL and Redis caching
- Integrated third-party APIs and built automated testing with pytest
- Worked in Agile/Scrum teams with 2-week sprints

EDUCATION
M.Tech in Computer Science — IIT Bombay (2017)

SKILLS
Python, Flask, Django, FastAPI, REST API, Microservices, PostgreSQL, MySQL, MongoDB, Redis, Docker, Kubernetes, AWS, Azure, Git, Linux, Scikit-learn, Pandas, Numpy, Machine Learning, NLP, CI/CD, Agile, Scrum

CERTIFICATIONS
AWS Certified Solutions Architect | Google Cloud Professional Data Engineer
"""
    },
    {
        "filename": "bob_sharma_mid_backend.pdf",
        "content": """
BOB SHARMA
bob.sharma@email.com | Mumbai, India

SUMMARY
Backend Developer with 4 years of experience in Python and Java. Strong in REST API development, SQL databases, and cloud deployments. Experience with Flask, Spring Boot, and Docker.

WORK EXPERIENCE

Backend Developer — CloudBase Systems (2021–Present)
- Built Flask REST APIs for e-commerce platform
- Worked with PostgreSQL and MySQL databases
- Deployed applications using Docker on AWS EC2
- Used Git for version control and code review

Junior Developer — StartupXYZ (2019–2021)
- Developed Python scripts for data processing
- Created SQL queries and stored procedures
- Participated in Agile sprints and daily standups

EDUCATION
B.Tech in Information Technology — University of Mumbai (2019)

TECHNICAL SKILLS
Python, Flask, Java, SQL, PostgreSQL, MySQL, Docker, AWS, Git, Linux, REST API, Agile

LANGUAGES
English (Fluent), Hindi (Native), Marathi (Native)
"""
    },
    {
        "filename": "carol_jones_junior_dev.pdf",
        "content": """
CAROL JONES
carol.jones@email.com | Pune, India

OBJECTIVE
Recent computer science graduate seeking a backend development position. Eager to learn and contribute to a dynamic team.

EDUCATION
B.Sc. Computer Science — Pune University (2023)
GPA: 8.2/10

INTERNSHIP
Python Developer Intern — TechCorp (June–August 2023)
- Developed a Flask web application as part of a team project
- Learned REST API fundamentals and worked with SQLite database
- Used Git for version control and submitted pull requests

PROJECTS
- Student Management System: Built using Python and Flask with SQLite
- Weather App: Used Python requests library to fetch weather API data
- Data Analysis: Analyzed CSV datasets using pandas and matplotlib

SKILLS
Python, Flask, SQL, SQLite, Git, HTML, CSS, JavaScript, Pandas, Matplotlib, REST API basics

COURSES COMPLETED
- Python for Data Science (Coursera)
- Web Development with Flask (Udemy)
- SQL for Beginners (edX)
"""
    },
    {
        "filename": "david_kumar_fullstack.pdf",
        "content": """
DAVID KUMAR
david.kumar@email.com | Bangalore, India | GitHub: davidkumar

PROFESSIONAL PROFILE
Full Stack Developer with 5 years of experience in Python, React, and Node.js. Skilled in building end-to-end web applications with modern tech stacks. Experience with microservices architecture, Docker, and cloud platforms.

EXPERIENCE

Full Stack Developer — InnovateTech (2020–Present)
- Developed Python Flask and Node.js REST APIs consumed by React frontend
- Built and maintained PostgreSQL and MongoDB databases
- Implemented Docker containerization and deployed on AWS
- Integrated machine learning models using scikit-learn and Flask APIs
- Used CI/CD pipelines and GitHub Actions for automated deployments

Software Developer — WebAgency Pro (2018–2020)
- Built responsive web applications with React and Django
- Developed RESTful APIs and integrated third-party services
- Worked in Scrum teams and participated in code reviews

EDUCATION
B.E. Computer Engineering — VTU (2018)

SKILLS
Python, Flask, Django, Node.js, React, JavaScript, TypeScript, REST API, PostgreSQL, MongoDB, Redis, Docker, AWS, GCP, Git, Linux, Scikit-learn, CI/CD, Agile, Scrum, HTML, CSS

ACHIEVEMENTS
- Reduced API response time by 40% through caching with Redis
- Led migration of monolith to microservices architecture
"""
    }
]

def create_pdf(resume_data):
    doc = SimpleDocTemplate(
        f"sample_resumes/{resume_data['filename']}",
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=1*inch,
        rightMargin=1*inch
    )
    styles = getSampleStyleSheet()
    story = []
    for line in resume_data['content'].strip().split('\n'):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    print(f"Created: sample_resumes/{resume_data['filename']}")

if __name__ == '__main__':
    import os
    os.makedirs('sample_resumes', exist_ok=True)
    try:
        from reportlab.lib.pagesizes import letter
        for r in resumes:
            create_pdf(r)
        print("\nAll sample resumes created in sample_resumes/ folder!")
        print("Upload these PDFs to test the Resume Ranker.")
    except ImportError:
        print("Install reportlab first: pip install reportlab")
