import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Preprocesses text by lowercasing, removing punctuation, and tokenizing."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def rank_resumes(job_description, resumes):
    """Ranks resumes based on their similarity to the job description."""

    processed_job_description = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]

    vectorizer = TfidfVectorizer()
    all_texts = [processed_job_description] + processed_resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    job_description_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()
    ranked_resumes = sorted(zip(similarities, resumes), key=lambda x: x[0], reverse=True)

    return ranked_resumes

# Example usage:
job_description = """
Software Engineer with experience in Python and machine learning.
Strong understanding of data structures and algorithms is required.
Experience with cloud platforms like AWS or Azure is a plus.
"""

resumes = [
    """
    John Doe
    Experience: Python, AWS, Machine Learning, Data Structures
    """,
    """
    Jane Smith
    Experience: Java, Web Development, SQL
    """,
    """
    Alice Johnson
    Experience: Python, Data Analysis, Statistics
    """,
    """
    Bob Williams
    Experience: Python, Machine Learning, algorithm development.
    """
]

ranked_resumes = rank_resumes(job_description, resumes)

print("Ranked Resumes:")
for similarity, resume in ranked_resumes:
    print(f"Similarity: {similarity:.4f}, Resume: {resume.splitlines()[0]}") # just print the first line of the resume.