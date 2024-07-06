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
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import csv
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        color: #FFFFFF;
    }
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
    }
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .big-font {
        font-size: 48px !important;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 30px;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .medium-font {
        font-size: 32px !important;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .small-font {font-size: 18px !important;}
    .highlight {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 6px solid #4B0082;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .highlight:hover {
        transform: translateY(-5px);
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
        color: #FFFFFF;
    }
    .stSelectbox {margin-top: 15px; margin-bottom: 15px;}
    .user-info {
        background-color: rgba(0, 0, 0, 0.7);
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
        background-color: rgba(0, 0, 0, 0.7);
        border: 1px solid #4B0082;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        color: #FFFFFF;
    }
    .stRadio>div {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 8px;
        padding: 10px;
    }
    .stExpander {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 8px;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.7);
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

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Add this new function to handle saving feedback
def save_feedback(feedback, rating):
    feedback_file = "Feedback.csv"
    feedback_exists = os.path.isfile(feedback_file)
    
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not feedback_exists:
            writer.writerow(["Timestamp", "Feedback", "Rating"])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, feedback, rating])

# Lottie animation
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_wd1udlcz.json"
lottie_json = load_lottieurl(lottie_url)

# Streamlit app
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<p class="big-font">🚀 Job Recommendation Dashboard</p>', unsafe_allow_html=True)

# Display Lottie animation
st_lottie(lottie_json, speed=0.5 , height=200, key="initial")

# User input
col1, col2 = st.columns(2)
with col1:
    input_type = st.radio("Select input type:", ("Applicant ID", "Skills and Qualifications"))

with col2:
    if input_type == "Applicant ID":
        user_id = st.text_input('Enter User ID (applicantId)', key='user_id')
    else:
        user_input = st.text_area("Enter your Skills and Qualifications", height=150)

if st.button('Get Recommendations'):
    if input_type == "Applicant ID" and user_id and user_id in user_data['applicantId'].values:
        user_info = user_data[user_data['applicantId'] == user_id].iloc[0]
        user_skills = user_info['combined_skills']
        
        # Display user information in sidebar
        st.sidebar.markdown('<p class="medium-font">User Information</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="medium-font">Top Job Recommendations:</p>', unsafe_allow_html=True)
        
        top_jobs = jobs.iloc[job_indices[:5]]
        
        # Create a radar chart for skills match
        skills = ['Python', 'Java', 'JavaScript', 'SQL', 'Machine Learning']
        user_skills_values = [user_skills.count(skill.lower()) for skill in skills]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_skills_values,
            theta=skills,
            fill='toself',
            name='Your Skills'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(user_skills_values)]
                )),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig)
        
        for i, (_, job) in enumerate(top_jobs.iterrows(), 1):
            st.markdown(f'<div class="highlight">', unsafe_allow_html=True)
            st.markdown(f'<p class="job-title">{i}. {job["jobTitle"]} - {job["client"]}</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<p class="job-detail"><strong>Position:</strong> {job["position"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="job-detail"><strong>Location:</strong> {job["location"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="job-detail"><strong>Job Type:</strong> {job.get("jobType", "Not Available")}</p>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<p class="job-detail"><strong>Skills:</strong> {job["skills"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="job-detail"><strong>Vacancies:</strong> {job["vacancies"]}</p>', unsafe_allow_html=True)
            
            with st.expander("View More Details"):
                st.markdown(f'<p class="job-detail"><strong>Status:</strong> {job.get("status", "Not Available")}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="job-detail"><strong>Recruiter:</strong> {job.get("recruiter", "Not Available")}</p>', unsafe_allow_html=True)
                
                # Display minimum required experience only if it's 10 years or less
                min_exp = job.get('minExp')
                if min_exp is not None and not pd.isna(min_exp):
                    try:
                        min_exp = float(min_exp)
                        if min_exp <= 10:
                            st.markdown(f'<p class="job-detail"><strong>Minimum Required Experience:</strong> {min_exp} years</p>', unsafe_allow_html=True)
                    except ValueError:
                        pass
                
                # Add concise description if valid
                if pd.notna(job['description']) and is_valid_description(job['description']):
                    description = job['description']
                    if len(description) > 200:
                        description = description[:200] + "..."
                    st.markdown(f'<p class="job-detail"><strong>Description:</strong> {description}</p>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No jobs available closely matching your input. Try broadening your search terms or consider adding relevant skills to your profile.")

st.markdown('</div>', unsafe_allow_html=True)



# Add some metrics or KPIs
st.markdown('<p class="medium-font">Dashboard Metrics</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Jobs", value=len(jobs), delta="5%")

with col2:
    st.metric(label="Active Users", value="1,234", delta="10%")

with col3:
    st.metric(label="Successful Placements", value="567", delta="15%")

# Add a chart showing job distribution by industry
industry_counts = jobs['client'].value_counts().head(10)

fig = go.Figure(data=[go.Bar(
    x=industry_counts.index,
    y=industry_counts.values,
    marker_color='#4B0082'
)])

fig.update_layout(
    title="Top 10 Industries with Job Openings",
    xaxis_title="Industry",
    yaxis_title="Number of Job Openings",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

st.plotly_chart(fig)


# Modify the feedback section
st.markdown('<p class="medium-font">User Feedback</p>', unsafe_allow_html=True)

feedback = st.text_area("Please provide your feedback on the job recommendations:", height=100)
rating = st.slider("Rate your experience (1-5 stars)", min_value=1, max_value=5, value=1)

if st.button("Submit Feedback"):
    if feedback.strip() != "":
        save_feedback(feedback, rating)
        st.success("Thank you for your feedback! We appreciate your input.")
    else:
        st.warning("Please enter some feedback before submitting.")



# Add a FAQ section
st.markdown('<p class="medium-font">Frequently Asked Questions</p>', unsafe_allow_html=True)

faq_data = [
    ("How are job recommendations generated?", "Our system uses advanced machine learning algorithms to match your skills and experience with available job openings."),
    ("Can I update my profile?", "Yes, you can update your profile by logging into your account and navigating to the 'Edit Profile' section."),
    ("How often are new jobs added?", "New job listings are added daily. We recommend checking back regularly for the latest opportunities."),
    ("What should I do if I find a job I'm interested in?", "Click on the job listing to view more details and follow the application instructions provided by the employer.")
]

for question, answer in faq_data:
    with st.expander(question):
        st.write(answer)

# End of the dashboard
st.markdown('<p class="small-font" style="text-align: center; margin-top: 50px;">Thank you for using our Job Recommendation Dashboard!</p>', unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 25px; background-color: rgba(0, 0, 0, 0.7); border-radius: 15px; box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);">
    <p style="color: #FFFFFF; font-size: 16px; font-weight: 500;">© 2024 Job Recommendation Dashboard. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)