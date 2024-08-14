import os
import openai
import pandas as pd
import streamlit as st
import joblib

# Set up the configuration for the application and OpenAI API key
class Config:
    OPENAI_API_KEY = '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'  # Replace with your actual OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY

# Handles sentiment analysis functionality
class SentimentAnalyzer:
    def __init__(self):
        # Load the sentiment analysis model and vectorizer
        self.model = joblib.load('sentiment_model.joblib')
        self.vectorizer = joblib.load('vectorizer.joblib')

    def predict_sentiment(self, text):
        vectorized_text = self.vectorizer.transform([text])
        prediction = self.model.predict(vectorized_text)[0]
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        return emotions[prediction]

# Manages chatbot functionalities including data summaries and generating responses
class Chatbot:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()

    def generate_summary_with_llm(data):
    summary_prompt = f"""
    You are a data-oriented mental health consultant with 16 years of experience in psychology and exceptional data analysis skills. Analyze the following monthly average Apple Watch health data spanning from 2020 to 2024. Use chain-of-thought reasoning to provide a comprehensive summary. Follow these steps:
    1. Identify significant trends and patterns in each metric over time. Explain your reasoning.
    2. Detect any anomalies and hypothesize potential causes. Show your thought process.
    3. Analyze correlations between different metrics. Describe how you arrived at these connections.
    4. Interpret findings through a psychological lens. Elaborate on your psychological reasoning.
    5. Propose data-driven, actionable recommendations for improving physical and mental well-being. Justify each recommendation.
    Present your analysis clearly, avoiding medical jargon where possible. Show your analytical rigor, creative insight, and empathetic understanding throughout your response.
    Data:
    {data}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI system designed to analyze health data with a focus on potential mental health implications. Approach this task with intellectual humility, recognizing the limits of data-driven insights in understanding human well-being."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=3000
    )
    return response['choices'][0]['message']['content'].strip()

    # Function to generate chatbot response
def generate_chatbot_response(user_data, user_message):
    analysis_prompt = f"""
        You are an AI assistant with extensive expertise in clinical psychology, health data analysis, and behavioral science. Your task is to perform a detailed psychological analysis of user data, focusing on holistic health and well-being. Approach this analysis as a seasoned clinical psychologist with years of experience in therapy and data-driven interventions.
        Analyze the following user data:

        Summary: {user_data['summary']}
        Sleep Quality: {user_data['sleep_quality']}
        Stress Level: {user_data['stress_level']}
        Goals: {','.join(user_data['goals'])}
        Feelings: {user_message}
        Sentiment: {user_data['sentiment']}

        Conduct a comprehensive psychological analysis using the following framework:

        Biopsychosocial Assessment:

        Biological factors: Analyze physical activity patterns, sleep quality, and potential physiological stress indicators.
        Psychological factors: Evaluate mood, cognitive patterns, emotional regulation, and self-efficacy.
        Social factors: Infer social support, work-life balance, and interpersonal stressors from available data.


        Cognitive-Behavioral Analysis:

        Identify cognitive distortions or maladaptive thought patterns.
        Analyze behavioral activation levels and avoidance behaviors.
        Assess the relationship between thoughts, feelings, and behaviors.


        Stress and Coping Mechanisms:

        Evaluate stress levels using transactional stress theory.
        Identify primary and secondary appraisal patterns.
        Analyze coping strategies (problem-focused vs. emotion-focused).


        Sleep Psychology:

        Assess sleep hygiene practices.
        Identify potential sleep disorders (insomnia, sleep apnea, etc.).
        Analyze the bidirectional relationship between sleep and mental health.


        Motivational Analysis:

        Evaluate intrinsic vs. extrinsic motivation for goals.
        Assess stage of change using the Transtheoretical Model.
        Identify potential barriers to goal achievement.


        Emotional Intelligence Assessment:

        Analyze emotional awareness and regulation.
        Evaluate empathy and social skills based on available data.


        Resilience and Well-being:

        Assess psychological resilience factors.
        Evaluate elements of well-being (PERMA model: Positive emotions, Engagement, Relationships, Meaning, Accomplishment).


        Personality Factors:

        Infer potential personality traits (Big Five) that may influence behavior patterns.


        Psychopathology Screening:

        Screen for potential mood disorders, anxiety disorders, or other mental health concerns.


        Intervention Planning:

        Develop a multi-modal intervention plan addressing identified issues.
        Include evidence-based therapies (CBT, ACT, DBT) where appropriate.
        Suggest lifestyle modifications to support mental and physical health.



        For each aspect of the analysis:

        Identify key psychological constructs and theories relevant to the data.
        Use clinical reasoning to form hypotheses about underlying psychological processes.
        Provide specific, actionable recommendations grounded in psychological research.
        Consider the interplay between different psychological and physiological factors.

        Your analysis should demonstrate:

        Deep understanding of clinical psychology principles and practices.
        Integration of data-driven insights with psychological theory.
        Holistic approach to mental and physical health.
        Empathetic yet professional tone.
        Clear, concise communication of complex psychological concepts.

        Conclude with a summary that:

        Highlights key findings and their psychological significance.
        Outlines a prioritized plan for addressing identified issues.
        Provides a prognosis based on the user's current state and potential for change.

        Remember to maintain ethical standards, acknowledging the limitations of remote analysis and emphasizing the importance of professional in-person assessment for definitive diagnoses or treatment plans.
        """

    analysis_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI system designed to analyze health data with a focus on potential mental health implications."},
            {"role": "user", "content": analysis_prompt}
        ],
        max_tokens=2000
    )
    
    analysis = analysis_response['choices'][0]['message']['content'].strip()

# Main Streamlit app class handling the UI and interactions
class StreamlitApp:
    def __init__(self):
        self.bot = Chatbot()
        self.setup_ui()

    def setup_ui(self):
        st.set_page_config(page_title="MindPal", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")
        st.markdown("""
            <style>
                .reportview-container { background: #e6f7ff; }
                .sidebar .sidebar-content { background: #f0f8ff; }
                .css-1v0mbdj {
                    width: 70%;
                    margin: 0 auto;
                    padding: 2rem;
                    border-radius: 10px;
                    background: white;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                }
                .stButton button { background: #4CAF50; color: white; }
            </style>
            """, unsafe_allow_html=True)
        self.manage_file_upload()

    def manage_file_upload(self):
        data_file = st.file_uploader("Upload your monthly average data CSV file", type=["csv"])
        if data_file:
            df = pd.read_csv(data_file)
            st.success("Data loaded successfully!")
            st.write(df)

            if 'summary' not in st.session_state:
                with st.spinner('Analyzing data and generating insights...'):
                    monthly_summary = self.bot.generate_summary_with_llm(df.to_string(index=False))
                st.session_state.summary = monthly_summary
                st.subheader("Data Analysis Summary")
                st.write(monthly_summary)

            self.collect_user_input()

    def collect_user_input(self):
        st.subheader("Personal Information")
        sleep_quality = st.slider("Rate your sleep quality (1-7)", 1, 7, value=4)
        stress_level = st.selectbox(
            "Rate your stress level",
            ["Never", "Rarely", "Occasionally", "Frequently", "Always"]
        )
        goals = st.multiselect("What are your main health goals?", ["Reduce stress", "Improve sleep", "Increase energy"])
        initial_feelings = st.text_area("Describe how you've been feeling lately:")

        if initial_feelings:
            sentiment = self.bot.analyzer.predict_sentiment(initial_feelings)
            st.write(f"Based on your input, it seems you might be feeling: {sentiment}")

            user_data = {
                'summary': st.session_state.summary,
                'sleep_quality': sleep_quality,
                'stress_level': stress_level,
                'goals': goals,
                'sentiment': sentiment
            }

            # Chat interface
            response = self.bot.generate_chatbot_response(user_data, initial_feelings)
            st.write("MindPal says:", response)

if __name__ == "__main__":
    Config()  # Load configuration
    app = StreamlitApp()  # Start the application
