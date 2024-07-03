import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Set page configuration
st.set_page_config(page_title="Job Recommendation Engine", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold;}
    .medium-font {font-size:20px !important; font-weight: bold;}
    .small-font {font-size:14px !important;}
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .job-title {color: #0066cc; font-size: 18px !important; font-weight: bold;}
    .job-detail {margin-left: 20px;}
    </style>
    """, unsafe_allow_html=True)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocess text data
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    if isinstance(text, str) and text.lower() != 'null' and text.strip() != '':
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        tokens = word_tokenize(text)
        filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)
    elif isinstance(text, (int, float)) and not pd.isna(text):
        return str(text)
    return ''

# Load data
@st.cache_data
def load_data():
    education = pd.read_csv('education.csv')
    employment = pd.read_csv('employment.csv')
    jobs = pd.read_csv('jobs.csv')
    return education, employment, jobs

education, employment, jobs = load_data()

# Merge education and employment data
user_data = pd.merge(education, employment, on='applicantId', how='outer')

# Combine skills and qualifications
edu_columns = ['primarySchool', 'secondarySchool', 'graduation', 'degree', 'fieldOfStudy']
emp_columns = ['currentDesignation', 'skills']
job_columns = ['jobTitle', 'position', 'skills', 'description']

user_data['combined_skills'] = user_data[edu_columns + emp_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
jobs['combined_skills'] = jobs[job_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

user_data['combined_skills'] = user_data['combined_skills'].apply(preprocess_text)
jobs['combined_skills'] = jobs['combined_skills'].apply(preprocess_text)

# Vectorize skills
vectorizer = TfidfVectorizer()
job_skills_tfidf = vectorizer.fit_transform(jobs['combined_skills'])

# Function to check if description is valid
def is_valid_description(description):
    if not isinstance(description, str):
        return False
    words = description.split()
    return len(words) >= 5 and len(set(words)) >= 3

# Function to get job recommendations
def get_job_recommendations(user_skills):
    user_tfidf = vectorizer.transform([user_skills])
    cosine_similarities = cosine_similarity(user_tfidf, job_skills_tfidf).flatten()
    job_indices = np.argsort(-cosine_similarities)
    return job_indices, cosine_similarities

# Streamlit app
st.markdown('<p class="big-font">Job Recommendation Engine</p>', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # User input
    input_type = st.radio("Select input type:", ("Applicant ID", "Skills and Qualifications"))
    
    if input_type == "Applicant ID":
        user_id = st.text_input('Enter User ID (applicantId)', key='user_id')
    else:
        user_input = st.text_area("Enter your Skills and Qualifications", height=100)

    if st.button('Get Recommendations'):
        if input_type == "Applicant ID" and user_id and user_id in user_data['applicantId'].values:
            user_info = user_data[user_data['applicantId'] == user_id].iloc[0]
            user_skills = user_info['combined_skills']
        elif input_type == "Skills and Qualifications" and user_input:
            user_skills = preprocess_text(user_input)
        else:
            st.error("Please enter valid input.")
            st.stop()
        
        job_indices, cosine_similarities = get_job_recommendations(user_skills)
        
        if np.any(cosine_similarities > 0.3):
            st.markdown('<p class="medium-font">Top Job Recommendations:</p>', unsafe_allow_html=True)
            for i, idx in enumerate(job_indices[:5], 1):
                job = jobs.iloc[idx]
                
                st.markdown(f'<div class="highlight">', unsafe_allow_html=True)
                st.markdown(f'<p class="job-title">{i}. {job["jobTitle"]} - {job["client"]}</p>', unsafe_allow_html=True)
                
                details = [
                    f'<p class="job-detail"><strong>Position:</strong> {job["position"]}</p>',
                    f'<p class="job-detail"><strong>Job Type:</strong> {job.get("jobType", "Not Available")}</p>',
                    f'<p class="job-detail"><strong>Location:</strong> {job["location"]}</p>',
                    f'<p class="job-detail"><strong>Status:</strong> {job.get("status", "Not Available")}</p>',
                    f'<p class="job-detail"><strong>Recruiter:</strong> {job.get("recruiter", "Not Available")}</p>',
                    f'<p class="job-detail"><strong>Skills:</strong> {job["skills"]}</p>',
                    f'<p class="job-detail"><strong>Vacancies:</strong> {job["vacancies"]}</p>'
                ]
                
                # Display minimum required experience only if it's 10 years or less
                min_exp = job.get('minExp')
                if min_exp is not None and not pd.isna(min_exp):
                    try:
                        min_exp = float(min_exp)
                        if min_exp <= 10:
                            details.append(f'<p class="job-detail"><strong>Minimum Required Experience:</strong> {min_exp} years</p>')
                    except ValueError:
                        pass
                
                # Add concise description if valid
                if pd.notna(job['description']) and is_valid_description(job['description']):
                    description = job['description']
                    if len(description) > 200:
                        description = description[:200] + "..."
                    details.append(f'<p class="job-detail"><strong>Description:</strong> {description}</p>')
                
                st.markdown('\n'.join(details), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No jobs available closely matching your input. Try broadening your search terms or consider adding relevant skills to your profile.")

with col2:
    if input_type == "Applicant ID" and user_id and user_id in user_data['applicantId'].values:
        st.markdown('<p class="medium-font">User Information</p>', unsafe_allow_html=True)
        user_info = user_data[user_data['applicantId'] == user_id].iloc[0]
        is_employed = pd.notna(user_info['employmentId'])
        
        if is_employed:
            st.markdown('<p class="small-font">Employment Details:</p>', unsafe_allow_html=True)
            for col in emp_columns + ['currentCompany', 'annualSalary']:
                if col in user_info and pd.notna(user_info[col]) and str(user_info[col]).lower() != 'null':
                    st.markdown(f"• {col.capitalize()}: {user_info[col]}")
        else:
            st.markdown('<p class="small-font highlight">Currently not employed</p>', unsafe_allow_html=True)