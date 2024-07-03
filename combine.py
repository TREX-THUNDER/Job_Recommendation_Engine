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
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #000000;
        color: #FFFFFF;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .big-font {
        font-size: 42px !important;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 30px;
        text-align: center;
    }
    .medium-font {
        font-size: 28px !important;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .small-font {font-size: 18px !important;}
    .highlight {
        background-color: #1A1A1A;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 6px solid #FFFFFF;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
    }
    .job-title {
        color: #FFFFFF;
        font-size: 24px !important;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .job-detail {
        margin-left: 25px;
        margin-bottom: 8px;
        font-size: 16px;
    }
    .stSelectbox {margin-top: 15px; margin-bottom: 15px;}
    .user-info {
        background-color: #1A1A1A;
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background-color: #4B0082;
        color: #FFFFFF;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #6A0DAD;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1A1A1A;
        border: 1px solid #FFFFFF;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        color: #FFFFFF;
    }
    .stRadio>div {
        background-color: #1A1A1A;
        border-radius: 8px;
        padding: 10px;
    }
    .stExpander {
        background-color: #1A1A1A;
        border-radius: 8px;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
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
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<p class="big-font" style="color: #4B0082;">🚀 Job Recommendation Engine</p>', unsafe_allow_html=True)

# User input
input_type = st.radio("Select input type:", ("Applicant ID", "Skills and Qualifications"))

if input_type == "Applicant ID":
    user_id = st.text_input('Enter User ID (applicantId)', key='user_id')
else:
    user_input = st.text_area("Enter your Skills and Qualifications", height=150)

if st.button('Get Recommendations'):
    if input_type == "Applicant ID" and user_id and user_id in user_data['applicantId'].values:
        user_info = user_data[user_data['applicantId'] == user_id].iloc[0]
        user_skills = user_info['combined_skills']
        
        # Display user information in sidebar
        st.sidebar.markdown('<p class="medium-font" style="color: #4B0082;">User Information</p>', unsafe_allow_html=True)
        is_employed = pd.notna(user_info['employmentId'])
        
        st.sidebar.markdown('<div class="user-info">', unsafe_allow_html=True)
        if is_employed:
            st.sidebar.markdown('<p class="small-font">Employment Details:</p>', unsafe_allow_html=True)
            for col in emp_columns + ['currentCompany', 'annualSalary']:
                if col in user_info and pd.notna(user_info[col]) and str(user_info[col]).lower() != 'null':
                    st.sidebar.markdown(f"• <strong>{col.capitalize()}:</strong> {user_info[col]}", unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<p class="small-font">Currently not employed</p>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    elif input_type == "Skills and Qualifications" and user_input:
        user_skills = preprocess_text(user_input)
    else:
        st.error("Please enter valid input.")
        st.stop()
    
    job_indices, cosine_similarities = get_job_recommendations(user_skills)
    
    if np.any(cosine_similarities > 0.3):
        st.markdown('<p class="medium-font" style="color: #4B0082;">Top Job Recommendations:</p>', unsafe_allow_html=True)
        
        top_jobs = jobs.iloc[job_indices[:5]]
        
        for i, (_, job) in enumerate(top_jobs.iterrows(), 1):
            st.markdown(f'<p class="job-title">{i}. {job["jobTitle"]} - {job["client"]}</p>', unsafe_allow_html=True)
            
            with st.expander("View Details"):
                st.markdown(f'<div class="highlight">', unsafe_allow_html=True)
                
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

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 25px; background-color: #1A1A1A; border-radius: 15px; box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);">
    <p style="color: #FFFFFF; font-size: 16px; font-weight: 500;">© 2024 Job Recommendation Engine. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)