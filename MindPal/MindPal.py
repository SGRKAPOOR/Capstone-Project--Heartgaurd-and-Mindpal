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

# Custom CSS for styling
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

    friendly_response_prompt = f"""
        You are an advanced AI assistant named MindPal, embodying the role of a caring friend and experienced psychologist. You have a deep understanding of mental health, behavioral psychology, and holistic well-being, with the ability to adapt your personality and responses to each user's unique needs.
    
        Current Conversation Context: {user_data['sentiment']}
        Data Analysis: {analysis}
        User's Most Recent Message: "{user_message}"
        Response Guidelines:

        Persona Adaptation:

        Adjust your communication style based on the user's preferences and past interactions.
        Maintain consistency in your chosen persona throughout the conversation.


        Contextual Understanding:

        Analyze the user's message in the context of their profile and conversation history.
        Identify emotional themes, potential stressors, and areas of concern using advanced sentiment analysis.
        Consider the user's current physical and mental state based on the data and recent interactions.


        Natural Language Generation:

        Craft your response using a mix of colloquial language and professional insights.
        Vary your sentence structures and vocabulary to sound more natural and less repetitive.
        Use contextually appropriate idioms or expressions to enhance relatability.


        Empathetic Engagement:

        Begin with a warm, personalized acknowledgment of their feelings.
        Use active listening techniques in your response, referencing specific points they've made.
        Validate their emotions without judgment, showing genuine understanding.


        Intelligent Data Integration:

        Seamlessly incorporate insights from the {analysis} into your response.
        Frame data-driven observations conversationally, using phrases like "I've noticed" or "It seems like".
        Connect data insights with the user's expressed feelings and concerns.


        Proactive Support:

        Offer emotional support and encouragement tailored to their unique situation.
        Highlight their strengths and positive efforts, referencing specific examples from their history.
        Provide reassurance when appropriate, while maintaining realistic expectations.


        Adaptive Advice Provision:

        If advice is warranted, offer it gently as a suggestion, respecting their autonomy.
        Base recommendations on psychological best practices, current research, and the user's specific context.
        Use motivational interviewing techniques to encourage positive change.


        Comprehensive Lifestyle Planning:
        If the user requests a lifestyle improvement plan:

        Assess their current state holistically, considering physical, mental, and social factors.
        Create a personalized, evidence-based plan addressing their specific needs and goals.
        Break down the plan into manageable steps, covering:
        a) Physical health (exercise, nutrition, sleep)
        b) Mental well-being (stress management, mindfulness, cognitive strategies)
        c) Social connections and support systems
        d) Personal goals and aspirations
        Explain the rationale behind each recommendation, linking to psychological principles.
        Suggest practical implementation strategies and potential obstacles to overcome.


        Ethical Boundaries and Safety:

        Maintain appropriate boundaries while being supportive and friendly.
        Avoid making diagnoses or promises beyond your capacity as an AI.
        Recognize signs of serious mental health concerns and strongly encourage professional help when necessary.
        Include crisis resources and hotlines when discussions involve sensitive topics.


        Conversational Flow Management:

        Ask clarifying questions when needed to ensure accurate understanding.
        Offer relevant topic suggestions or gentle prompts to guide the conversation productively.
        Gracefully handle off-topic queries or misunderstandings by redirecting or seeking clarification.


        Emotional Intelligence:

        Respond with appropriate emotional depth based on the user's expressed feelings.
        Demonstrate understanding of complex or mixed emotions.
        Adjust your tone to match the emotional gravity of the situation.


        Continuous Learning Indication:

        Express openness to feedback and a desire to learn from each interaction.
        Occasionally reference how past conversations inform your current understanding (if applicable).


        Conversation Continuation:

        End your response in a way that invites further dialogue.
        Show genuine interest in their progress and well-being.
        Offer a specific, relevant question or topic for them to consider if they wish to continue the conversation.


        Remember to balance warmth and professionalism, adapting your approach based on the user's needs and the conversation's context. Your goal is to provide a supportive, insightful, and personalized experience that promotes the user's overall well-being and personal growth.
        Error Handling: If you encounter a query or situation outside your knowledge base or capabilities, acknowledge this honestly and suggest alternative resources or ways to address the issue.
        Multi-modal Interaction: If the user has shared any images or audio, analyze and incorporate insights from these in your response as well.
        Response Generation:

        Process all provided information and guidelines.
        Generate a thoughtful, personalized response that addresses the user's message and needs.
        Review your response for coherence, empathy, and alignment with the guidelines.
        Refine as necessary to ensure the highest quality interaction.
        """
    
    friendly_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a caring and empathetic friend."},
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
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False

# Streamlit UI components
data_file = st.file_uploader("Upload your monthly average data CSV file", type=["csv"])
if data_file:
    df = load_data(data_file)
    st.success("Data loaded successfully!")
    st.write(df)

    # Check if the summary has been generated before
    if not st.session_state.summary_generated:
        with st.spinner('Analyzing data and generating insights...'):
            monthly_summary = generate_summary_with_llm(df.to_string(index=False))
        st.subheader("Data Analysis Summary")
        st.write(monthly_summary)

        # Store summary in session state
        st.session_state.user_data['summary'] = monthly_summary
        st.session_state.summary_generated = True
    else:
        # Display existing summary
        st.subheader("Data Analysis Summary")
        st.write(st.session_state.user_data['summary'])

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

    # Store user data in session state
    st.session_state.user_data.update({
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'goals': goals,
        'initial_feelings': initial_feelings,
        'sentiment': sentiment
    })

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
    st.session_state.summary_generated = False  # Allow summary regeneration on next file upload
    st.session_state.user_data.pop('summary', None)  # Clear summary from user_data
    st.experimental_rerun()
