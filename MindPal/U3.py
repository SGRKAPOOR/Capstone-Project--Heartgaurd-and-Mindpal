import os
import openai
import pandas as pd
import streamlit as st
import joblib

# Set your OpenAI API key here
openai_api_key = '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'
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Load the sentiment analysis model and vectorizer
sentiment_model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Function to predict sentiment
def predict_sentiment(text):
    vectorized_text = vectorizer.transform([text])
    prediction = sentiment_model.predict(vectorized_text)[0]
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return emotions[prediction]

# Streamlit page configuration
st.set_page_config(
    page_title="MindPal",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (same as before)
st.markdown("""
    <style>
        .reportview-container {
            background: #e6f7ff;
        }
        .sidebar .sidebar-content {
            background: #f0f8ff;
        }
        .css-1v0mbdj {
            width: 70%;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            background: white;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            background: #4CAF50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center; color: #333;'>MindPal</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center; color: #666;'>Your friendly chatbot for personalized mental health support. Please upload your data file to get started.</p>", unsafe_allow_html=True)

# Function to load data from CSV with caching
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Enhanced function to generate summaries and handle incomplete responses
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
    Analyze the following user data:
    
    * Summary: {user_data['summary']}
    * Sleep Quality: {user_data['sleep_quality']}
    * Stress Level: {user_data['stress_level']}
    * Goals: {', '.join(user_data['goals'])}
    * Feelings: {user_message}
    * Sentiment: {user_data['sentiment']}

    Weights and Rationales:
    * Summary (Weight: 20%): This data provides a broad overview of the user's physical activity and can be crucial for understanding long-term trends in physical health, which are important from both a data science and psychological perspective.
    * Sleep Quality (Weight: 20%): Sleep quality is a critical factor in mental and physical health. A psychologist would likely place significant emphasis on sleep patterns, and a data scientist would value the quantifiable nature of this data.
    * Stress Level (Weight: 20%): Stress levels can significantly impact both mental state and physical health. This data point is valuable for making informed decisions and personalized recommendations, aligning well with psychological and data-analytic expertise.
    * Goals (Weight: 15%): Goals give context to the data and help personalize the responses. Understanding user goals is critical from a psychological perspective to provide tailored advice and from a data science perspective to align insights with user expectations.
    * Feelings (Weight: 15%): The user's feelings are essential for psychological analysis and for tailoring interactions in a manner that respects the user's current emotional state. This input helps the model adjust its responses to be more empathetic and relevant.
    * Sentiment (Weight: 10%): This provides a quick, automated insight into the user's emotional tone, which is useful for adjusting the interaction tone. While it's a derivative of the feelings data, it helps reinforce the analysis with quantified sentiment, valuable in data-driven psychological assessments.

    Perform a detailed analysis following these steps:
    1. Identify and explain trends and patterns in the user's physical activity.
    2. Detect anomalies and hypothesize causes, particularly in sleep quality and stress levels.
    3. Analyze correlations between different metrics.
    4. Interpret findings from a psychological perspective.
    5. Provide actionable recommendations based on the user's goals.

    Provide a concise summary of your analysis.
    """

    analysis_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI system designed to analyze health data with a focus on potential mental health implications."},
            {"role": "user", "content": analysis_prompt}
        ],
        max_tokens=500
    )
    
    analysis = analysis_response['choices'][0]['message']['content'].strip()

    friendly_response_prompt = f"""
    You are Friend with expertise in psychology, health data analysis, and interpersonal relationships. Your primary function is to interpret user health data and provide brief, targeted responses. You embody key friendship qualities: trustworthiness, empathy, supportiveness, good communication, and reliability.

    An analysis of the user's health data shows:
    {analysis}

    The user's most recent message and sentiment:
    "{user_message}",  "{user_data['sentiment']} "

    Response Guidelines:

    ### If the user specifically requests a plan for lifestyle improvement,making their physical and mental health better:
    Provide a detailed plan covering the following areas:
    - **Diet and Nutrition**
    - **Exercise and Physical Activity**
    - **Sleep and Relaxation**
    - **Mental Well-being and Stress Management**
    - **Time Management and Productivity**

    For each area, offer 3-5 actionable recommendations based on psychological best practices and the user's data analysis. Communicate these clearly and reliably, maintaining a professional yet friendly tone.

    ### Else, if the user asks a general query or makes a comment:
    Provide a concise, data-driven response based on the {analysis}. Be a psychologist friend in a judgement environment, highlighting the most relevant information. Demonstrate empathy, fun and supportiveness in your tone.

    ### Additionally:
    - If the user expresses strong emotions or concerns, acknowledge them empathetically before focusing on the data-based insights or recommendations.
    - Use phrases to emphasize the data-driven nature of your responses, demonstrating trustworthiness.
    - Avoid lengthy explanations or anecdotes unless specifically requested by the user.
    - Consistently provide accurate, helpful information based on the user's data and queries."""


    friendly_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a caring and empathetic friend with 15 years of clinical psychologist."},
            {"role": "user", "content": friendly_response_prompt}
        ],
        max_tokens=500
    )

    return friendly_response['choices'][0]['message']['content'].strip()

# Initialize session state for chat history and user data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Streamlit UI components
data_file = st.file_uploader("Upload your monthly average data CSV file", type=["csv"])
if data_file:
    df = load_data(data_file)
    st.success("Data loaded successfully!")
    st.write(df)

    with st.spinner('Analyzing data and generating insights...'):
        monthly_summary = generate_summary_with_llm(df.to_string(index=False))
    st.subheader("Data Analysis Summary")
    st.write(monthly_summary)

    # Additional user inputs for personalized advice
    st.subheader("Personal Information")
    sleep_quality = st.slider("Rate your sleep quality (1-7)", 1, 7, value=4)
    stress_level = st.selectbox(
        "Rate your stress level",
        options=[
            "Never",
            "Rarely (less than once a week)",
            "Occasionally (1-2 times a week)",
            "Frequently (3-4 times a week)",
            "Almost always (daily)"
        ]
    )
    goals = st.multiselect("What are your main health goals?", ["Reduce stress", "Improve sleep", "Increase energy", "Better mood", "Weight management"])
    initial_feelings = st.text_area("Describe how you've been feeling lately:")

    # Analyze user sentiments
    sentiment = predict_sentiment(initial_feelings)
    st.write(f"Based on your input, it seems you might be feeling: {sentiment}")
    examples = """#### User Requesting a Lifestyle Improvement Plan:
    User's most recent message:
    "I need help improving my lifestyle. Can you provide a detailed plan?"

    Response:
    "Based on your data, here’s a detailed plan to improve your lifestyle:

    a) **Diet and Nutrition**:
    - Incorporate more fruits, vegetables, and whole grains into your meals.
    - Stay hydrated by drinking at least 8 glasses of water daily.
    - Reduce processed foods and sugars.

    b) **Exercise and Physical Activity**:
    - Aim for at least 30 minutes of moderate exercise most days of the week, such as walking or cycling.
    - Include strength training exercises twice a week.
    - Try to incorporate more movement into your daily routine, like taking the stairs.

    c) **Sleep and Relaxation**:
    - Maintain a consistent sleep schedule by going to bed and waking up at the same time every day.
    - Develop a relaxing bedtime routine, such as reading or listening to calm music.
    - Practice relaxation techniques like deep breathing or meditation before bed.

    d) **Mental Well-being and Stress Management**:
    - Take regular breaks throughout the day to manage stress.
    - Engage in activities you enjoy to relax and unwind.
    - Consider talking to a mental health professional if needed.

    e) **Time Management and Productivity**:
    - Create a daily schedule to manage your tasks efficiently.
    - Set realistic goals and prioritize them.
    - Avoid multitasking to enhance focus and productivity.

    #### User Making a General Comment:
    User's most recent message:
    "I'm feeling a bit off lately and not sure why."

    Response:
    "Based on your data, it seems like your sleep quality has been inconsistent, which might be affecting your mood. Ensuring a regular sleep schedule and practicing relaxation techniques before bed could help improve how you’re feeling. Remember, I’m here for you."""

    # Store user data in session state
    st.session_state.user_data = {
        'summary': monthly_summary,
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'goals': goals,
        'initial_feelings': initial_feelings,
        'sentiment': sentiment,
        'examples' : examples
    
    }

    # Chat interface
    st.subheader("Chat with Your MindPal")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the chatbot function
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = generate_chatbot_response(st.session_state.user_data, prompt)
            message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    st.warning("Please upload a CSV file to proceed.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()