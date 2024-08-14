import os
import openai
import pandas as pd
import streamlit as st
from langchain_community.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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
    page_title="Psychological Friend",
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
st.markdown("<h1 style='text-align: center; color: #333;'>Psychological Friend</h1>", unsafe_allow_html=True)
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



        Remember to balance warmth and professionalism, adapting your approach based on the user's needs and the conversation's context. Your goal is to provide a supportive, insightful, and personalizedexperience that promotes the user's overall well-being and personal growth.
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

    # Selectbox for stress level with categorical frequency
    stress_level = st.selectbox(
        "Rate your stress level",
        options=[
            "Never (0)",
            "Rarely (less than once a week) (1)",
            "Occasionally (1-2 times a week) (2)",
            "Frequently (3-4 times a week) (3)",
            "Almost always (daily) (4)"
        ]
    )

    goals = st.multiselect("What are your main health goals?", ["Reduce stress", "Improve sleep", "Increase energy", "Better mood", "Weight management"])
    feelings = st.text_area("Describe how you've been feeling lately:")

    # Analyze user sentiments
    sentiment = predict_sentiment(feelings)
    st.write(f"Based on your input, it seems you might be feeling: {sentiment}")

    weights = {
        "summary": {
            "weight": 0.20,
            "rationale": "This data provides a broad overview of the user's physical activity and can be crucial for understanding long-term trends in physical health, which are important from both a data science and psychological perspective."
        },
        "sleep_quality": {
            "weight": 0.20,
            "rationale": "Sleep quality is a critical factor in mental and physical health. A psychologist would likely place significant emphasis on sleep patterns, and a data scientist would value the quantifiable nature of this data."
        },
        "stress_level": {
            "weight": 0.20,
            "rationale": "Stress levels can significantly impact both mental state and physical health. This data point is valuable for making informed decisions and personalized recommendations, aligning well with psychological and data-analytic expertise."
        },
        "goals": {
            "weight": 0.15,
            "rationale": "Goals give context to the data and help personalize the responses. Understanding user goals is critical from a psychological perspective to provide tailored advice and from a data science perspective to align insights with user expectations."
        },
        "feelings": {
            "weight": 0.15,
            "rationale": "The user's feelings are essential for psychological analysis and for tailoring interactions in a manner that respects the user's current emotional state. This input helps the model adjust its responses to be more empathetic and relevant."
        },
        "sentiment": {
            "weight": 0.10,
            "rationale": "This provides a quick, automated insight into the user's emotional tone, which is useful for adjusting the interaction tone. While it's a derivative of the feelings data, it helps reinforce the analysis with quantified sentiment, valuable in data-driven psychological assessments."
        }
    }

    # Prompt Templates with advanced chain-of-thought prompting
    first_input_prompt = PromptTemplate(
        input_variables=['summary', 'sleep_quality', 'stress_level', 'goals', 'feelings', 'sentiment', 'weights'],
        template="""
        You are a chatbot with expertise in data science and psychology. Your responses should be structured to provide insightful, empathetic, and contextually appropriate analysis based on the user's input data. Use the following weighting scheme to prioritize different types of data in your responses:

        Personal info:
        - Sleep quality: {sleep_quality} out of a score of seven
        - Stress level: User has been feeling {stress_level} this frequently
        - Health goals: {goals}
        - Recent feelings: {feelings}
        - Feeling sentiment: {sentiment}

        For each user interaction, follow this structured response framework with the assigned {weights} for the analysis and provide me Summary:

        1. Introduction/Context: Acknowledge the user's input contextually.
        2. Data Insight: Provide specific insights based on the weighted importance of the data inputs and go through the {summary} to give an analysis.
        3. Psychological Insight: Offer a psychological perspective relevant to the user’s emotional and mental state as indicated by their feelings and sentiment analysis.
        4. Actionable Advice: Suggest practical steps or changes based on the combined insights from data and psychological analysis.
        5. Feedback Solicitation: Encourage ongoing dialogue and data sharing to refine further interactions and advice.
        6. Closing: Conclude with a motivational statement that reinforces the chatbot’s supportive role.

        Use this structured framework and weighting scheme to ensure each response leverages both psychological and data science principles effectively. And provide a summary for the whole analysis.
        """
    )

    second_input_prompt = PromptTemplate(
        input_variables=['description'],
        template="""
        Based on your previous analysis: {description}

        Now, let's dig a little deeper. As a friendly, approachable psychologist, use your expertise to provide insights in the following structured format (about 300 words):

        **Specific Mental Health Symptoms**
        - Reference DSM-5 criteria and explain them in simple terms.
        - Include nuanced clinical observations beyond textbook definitions.

        **Connections to Data and Personal Info**
        - Highlight how symptoms relate to the data and personal info.
        - Identify any "aha!" moments or subtle connections.

        **Causes of Symptoms**
        - Explain possible causes using psychological theories (e.g., attachment theory, stress-diathesis model, cognitive schemas).
        - Discuss how reported goals and feelings may be manifestations of deeper psychological processes.

        **Urgency and Attention Needed**
        - Identify symptoms that are urgent or require immediate attention.
        - Explain why, considering acute risks and long-term impacts.

        **Influence of Background**
        - Discuss how cultural, social, and environmental backgrounds may influence symptom presentation.

        **Perspective and Interpretation**
        - Reflect on how your perspective as a psychologist shapes your interpretation.
        - Consider alternative viewpoints.

        Use relatable analogies or examples to make complex concepts easy to understand. Your goal is to provide a thoughtful analysis that feels like an enlightening conversation rather than a clinical report.
        """
    )

    third_input_prompt = PromptTemplate(
        input_variables=['symptoms'],
        template="""
        Alright, given these symptoms: {symptoms}

        Time for some fun, evidence-based problem-solving! As a friendly psychologist, provide engaging advice in the following structured format (about 300 words):

        **Coping Strategies and Interventions**
        - Suggest 3-5 strategies from various therapeutic approaches (e.g., CBT, DBT, ACT, mindfulness, psychodynamic).

        **Explanation and Analogies**
        - Explain why each strategy might work using quirky analogies or everyday examples.
        - Mention relevant psychological research or theories.

        **Fit with Current Habits and Goals**
        - Discuss how these strategies align with the person's current habits, goals, and data trends.
        - Identify potential clashes or challenges.

        **Prioritization**
        - Recommend which strategies to try first and explain why.
        - Consider the individual's resources and readiness for change.

        **Engagement and Integration**
        - Suggest fun, engaging ways to integrate these interventions into daily routines.

        **Addressing Symptoms and Processes**
        - Explain how each intervention addresses both surface-level symptoms and underlying psychological processes.

        **Potential Risks**
        - Mention any potential risks or contraindications lightly but clearly.

        Use practical, engaging language and present these interventions as exciting opportunities for growth rather than clinical treatments. Think of yourself as a friendly life coach with a solid grounding in psychology.
        """
    )

    fourth_input_prompt = PromptTemplate(
        input_variables=['interventions'],
        template="""
        Great! We've got these interventions: {interventions}

        Now, let's make this happen! As a friendly psychologist with extensive experience, craft an exciting implementation plan in the following structured format (about 300 words):

        **Daily Plan**
        - Sketch out a fun, doable daily plan. Think "mental health makeover adventure" rather than "boring routine".
        - Consider the individual's unique circumstances.

        **Overcoming Obstacles**
        - Identify potential obstacles and suggest creative solutions to overcome them.
        - Draw from motivational interviewing and stages of change models.

        **Support Systems and Resources**
        - Identify support systems and resources to make this journey easier and more enjoyable.
        - Include professional services and community resources, framing them as "adventure gear".

        **Tracking Progress**
        - Propose creative ways to track progress that feel more like fun check-ins than chores.
        - Incorporate both objective measures and subjective experiences.

        **Maintaining Motivation**
        - Suggest ways to foster self-efficacy and intrinsic motivation.
        - Focus on making the individual want to continue because of the benefits.

        **Using Health Data**
        - Recommend ways to use ongoing health data to celebrate wins and fine-tune the plan.
        - Create a positive feedback loop to maintain excitement and progress.

        **Supportive Alliance**
        - Briefly touch on how to maintain a supportive alliance through the process, even in a digital/remote context.
        - Stay connected and encouraging as their friendly psychologist guide.

        Use friendly, practical, and upbeat language. Make the mental health improvement journey feel like an exciting adventure that the individual is eager to embark on. Use encouraging language, fun metaphors, and convey confidence in their ability to succeed!
        """
    )

    # Initialize and apply Langchain for advanced chain-of-thought prompting
    llm = LangchainOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='description')
    chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='symptoms')
    chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='interventions')
    chain4 = LLMChain(llm=llm, prompt=fourth_input_prompt, verbose=True, output_key='implementation')
    Parent_Chain = SequentialChain(
        chains=[chain1, chain2, chain3, chain4],
        input_variables=['summary', 'sleep_quality', 'stress_level', 'goals', 'feelings', 'sentiment', 'weights'],
        output_variables=['description', 'symptoms', 'interventions', 'implementation'],
        verbose=True
    )

    if feelings:
        with st.spinner('Analyzing your information and generating personalized advice...'):
            response = Parent_Chain({
                'summary': monthly_summary, 
                'sleep_quality': sleep_quality, 
                'stress_level': stress_level, 
                'goals': ', '.join(goals),
                'feelings': feelings,
                'sentiment': sentiment,
                'weights': weights
            })
            
            st.subheader("Personalized Mental Health Analysis:")
            st.text_area("Description", response['description'], height=500)
else:
    st.warning("Please upload a CSV file to proceed.")
