import os
import openai
import pandas as pd
import streamlit as st
import random
from langchain_community.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set your OpenAI API key here
openai_api_key = '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'
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Streamlit framework
st.title('Advanced Mental Health Solutions')

# Load monthly average data from CSV
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to create a summary using OpenAI API with chain-of-thought prompting
def generate_summary_with_llm(data):
    summary_prompt = (
        "You are a data-oriented mental health consultant with 16 years of experience in psychology and exceptional data analysis skills. "
        "Analyze the following monthly average Apple Watch health data spanning from 2020 to 2024. "
        "Use chain-of-thought reasoning to provide a comprehensive summary. Follow these steps:\n\n"
        "1. Identify significant trends and patterns in each metric over time. Explain your reasoning.\n"
        "2. Detect any anomalies and hypothesize potential causes. Show your thought process.\n"
        "3. Analyze correlations between different metrics. Describe how you arrived at these connections.\n"
        "4. Interpret findings through a psychological lens. Elaborate on your psychological reasoning.\n"
        "5. Propose data-driven, actionable recommendations for improving physical and mental well-being. Justify each recommendation.\n"
        "6. Identify areas where more data or context would be beneficial. Explain why this additional information would be valuable.\n"
        "7. Discuss potential limitations of the analysis. Reflect on how these limitations impact your conclusions.\n\n"
        "Present your analysis clearly, avoiding medical jargon where possible. Show your analytical rigor, creative insight, and empathetic understanding throughout your response.\n\n"
        f"Data:\n{data}"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI system designed to analyze health data with a focus on potential mental health implications. Approach this task with intellectual humility, recognizing the limits of data-driven insights in understanding human well-being."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=1500
    )
    
    return response.choices[0].message['content'].strip()

# File uploader
data_file = st.file_uploader("Upload your monthly average data CSV file", type=["csv"])

if data_file:
    df = load_data(data_file)
    st.write("Data loaded successfully!")
    st.write(df)

    # Generate summary from data using OpenAI API
    with st.spinner('Analyzing data and generating insights...'):
        monthly_summary = generate_summary_with_llm(df.to_string(index=False))
    st.subheader("Data Analysis Summary:")
    st.write(monthly_summary)

    # Additional user inputs
    st.subheader("Personal Information")
    sleep_quality = st.slider("Rate your sleep quality (1-10)", 1, 10)
    stress_level = st.slider("Rate your stress level (1-10)", 1, 10)
    exercise_frequency = st.selectbox("How often do you exercise?", 
                                      ["Rarely", "1-2 times/week", "3-4 times/week", "5+ times/week"])
    goals = st.multiselect("What are your main health goals?", 
                           ["Reduce stress", "Improve sleep", "Increase energy", "Better mood", "Weight management"])
    
    input_text = st.text_area("Describe how you've been feeling lately:")

    # Prompt Templates with advanced chain-of-thought prompting

first_input_prompt = PromptTemplate(
    input_variables=['summary', 'sleep_quality', 'stress_level', 'exercise_frequency', 'goals', 'feelings'],
    template="""You're a friendly psychologist with 16 years of experience and a knack for explaining complex ideas simply. Analyze this info with a smile:

Data summary: {summary}

Personal info:
Sleep quality: {sleep_quality}/10
Stress level: {stress_level}/10
Exercise frequency: {exercise_frequency}
Health goals: {goals}
Recent feelings: {feelings}


In about 300 words, address the following:

1. Identify key trends in the data and personal info. How might physiological patterns be influencing psychological states and vice versa? Any surprising connections?

2. Spot potential mental health concerns based on this analysis. Draw clear links between physical health metrics and possible psychological implications.

3. Using a mix of psychodynamic and cognitive-behavioral frameworks, interpret the person's reported feelings and goals. How might past experiences or thought patterns be shaping their current state?

4. Examine how lifestyle factors (sleep, exercise, stress) could be impacting overall mental well-being. What's the relationship between their habits and mood?

5. Synthesize these perspectives to form a clear picture of their current mental health status. What's your overall impression?

6. What additional information would be helpful to have, and why? How would this extra data enhance our understanding?

Keep your response friendly and clear - imagine you're explaining this to a curious friend over coffee! Use relatable examples or analogies where possible to make your points more engaging and memorable."""
)


second_input_prompt = PromptTemplate(
    input_variables=['description'],
    template="""Based on your previous analysis: {description}

Now, let's dig a little deeper. As a friendly, approachable psychologist, use your expertise to provide insights in about 300 words:

1. What specific mental health symptoms might be present? Reference DSM-5 criteria, but explain it like you're chatting over coffee. Include any nuanced clinical observations that go beyond textbook definitions.

2. How do these symptoms connect to the data and personal info? Highlight any "aha!" moments or subtle connections that might not be immediately obvious.

3. What might be causing these symptoms? Get creative in your explanations, but stay grounded in psychological theories (e.g., attachment theory, stress-diathesis model, cognitive schemas). How might the individual's reported goals and feelings be manifestations of deeper psychological processes or coping mechanisms?

4. Are any of these symptoms urgent or requiring immediate attention? Explain why, considering both acute risks and potential long-term impacts.

5. How might the person's cultural, social, and environmental background be influencing the presentation of these symptoms? Consider factors that might not be explicitly stated in the data.

6. Briefly reflect on how your own perspective as a psychologist might be shaping your interpretation. Are there alternative viewpoints worth considering?

Remember, we're aiming for friendly and insightful - like a wise, approachable mentor! Use relatable analogies or examples to make complex concepts easy to understand. Your goal is to provide a thoughtful analysis that feels like an enlightening conversation rather than a clinical report."""
)

third_input_prompt = PromptTemplate(
    input_variables=['symptoms'],
    template="""Alright, given these symptoms: {symptoms}

Time for some fun, evidence-based problem-solving! As a friendly psychologist, provide engaging advice in about 300 words:

1. Suggest 3-5 coping strategies or interventions, drawing from various therapeutic approaches (e.g., CBT, DBT, ACT, mindfulness, psychodynamic). Mix it up creatively!

2. Explain why each strategy might work, using quirky analogies or everyday examples. Briefly mention relevant psychological research or theories to back up each approach.

3. How do these strategies fit with the person's current habits, goals, and physiological data trends? Identify any potential clashes or challenges.

4. Prioritize these strategies. Which should they try first, and why? Consider the individual's resources and readiness for change.

5. How can we make these strategies feel less like "homework" and more like exciting "life upgrades"? Suggest fun, engaging ways to integrate these interventions into daily routines.

6. Briefly touch on how each intervention addresses both surface-level symptoms and underlying psychological processes. Use simple, relatable language.

7. Any potential risks or contraindications to watch out for? Mention these lightly but clearly.

Remember, we're aiming for practical, engaging, and even a little entertaining! Your goal is to present these interventions as exciting opportunities for growth rather than clinical treatments. Think of yourself as a friendly life coach with a solid grounding in psychology."""
)


fourth_input_prompt = PromptTemplate(
    input_variables=['interventions'],
    template="""Great! We've got these interventions: {interventions}

Now, let's make this happen! As a friendly psychologist with extensive experience, craft an exciting implementation plan in about 300 words:

1. Sketch out a fun, doable daily plan. Think "mental health makeover adventure" rather than "boring routine". Consider the individual's unique circumstances and how to weave interventions seamlessly into their life.

2. What obstacles might pop up on this journey? How can we creatively "ninja" our way around them? Draw from motivational interviewing and stages of change models to suggest solutions that feel empowering rather than challenging.

3. Identify support systems and resources to make this journey easier and more enjoyable. Include both professional services and community resources, framing them as "adventure gear" for this mental health quest.

4. Propose creative ways to track progress that feel more like fun check-ins than chores. How can we incorporate both objective measures and subjective experiences in a way that's engaging?

5. How do we keep the motivation flowing? Focus on fostering self-efficacy and intrinsic motivation. Think less "you should do this" and more "you'll want to do this because...".

6. Suggest ways to use ongoing health data to celebrate wins and fine-tune the plan. How can we create a positive feedback loop that makes the individual excited to continue their progress?

7. Briefly touch on how to maintain a supportive alliance through this process, even in a digital/remote context. How can you, as their friendly psychologist guide, stay connected and encouraging?

Remember, we're going for friendly, practical, and upbeat. Your goal is to make this mental health improvement journey feel like an exciting adventure that the individual is eager to embark on. Use encouraging language, fun metaphors, and a tone that conveys your confidence in their ability to succeed!"""
)


# OpenAI LLM with API key
llm = LangchainOpenAI(temperature=0.7, openai_api_key=openai_api_key)

# Create chains
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='description')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='symptoms')
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='interventions')
chain4 = LLMChain(llm=llm, prompt=fourth_input_prompt, verbose=True, output_key='implementation')

# Sequential Chain
Parent_Chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=['summary', 'sleep_quality', 'stress_level', 'exercise_frequency', 'goals', 'feelings'],
    output_variables=['description', 'symptoms', 'interventions', 'implementation'],
    verbose=True
)

# Execute chain if input_text is provided   
if input_text:
    with st.spinner('Analyzing your information and generating personalized advice...'):
        response = Parent_Chain({
            'summary': monthly_summary, 
            'sleep_quality': sleep_quality, 
            'stress_level': stress_level, 
            'exercise_frequency': exercise_frequency, 
            'goals': ', '.join(goals),
            'feelings': input_text
        })
    
    st.subheader("Personalized Mental Health Analysis:")
    st.write(response['description'])
    st.subheader("Potential Symptoms and Concerns:")
    st.write(response['symptoms'])
    st.subheader("Recommended Interventions:")
    st.write(response['interventions'])
    st.subheader("Implementation Strategies:")
    st.write(response['implementation'])

    # Add a random Friends quote
    friends_quotes = [
        "How you doin'?",
        "We were on a break!",
        "It's a moo point. It's like a cow's opinion. It just doesn't matter.",
        "It's like all of my life, everyone has always told me, 'You're a shoe!' Well, what if I don't want to be a shoe?",
        "Pivot! Pivot! Pivot!",
        "Smelly cat, smelly cat, what are they feeding you?",
        "I'm a lot of things, but 'crazy' is not one of them. I'm a friend, a boss, a leader, a mentor... but not crazy.",
        "I'm not great at the advice. Can I interest you in a sarcastic comment?",
        "I'm not a vegetarian because I love animals. I'm a vegetarian because I hate plants.",
        "It's like, I'm not even a real person, I'm just a character in a TV show.",
        "I'm not stupid, I'm just... creatively challenged."
    ]
    st.write("\n\nAnd remember, as they say in Friends:")
    st.write(f"*\"{random.choice(friends_quotes)}\"*")
else:
    st.write("Please upload a CSV file to proceed.")
