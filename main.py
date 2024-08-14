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
        template="""You are a psychologist with 16 years of experience, specializing in integrative approaches that combine cognitive-behavioral, psychodynamic, and holistic methodologies. Based on the following data summary and personal information, provide a comprehensive mental health analysis. Use detailed chain-of-thought reasoning to explain your process:

Data summary: {summary}

Personal info:
Sleep quality: {sleep_quality}/10
Stress level: {stress_level}/10
Exercise frequency: {exercise_frequency}
Health goals: {goals}
Recent feelings: {feelings}

Steps:
1. Analyze the data trends in relation to the personal information. Consider how physiological patterns might be influencing psychological states and vice versa.
2. Identify potential mental health concerns based on this analysis. Draw connections between physical health metrics and potential psychological implications.
3. Apply psychodynamic principles to interpret the individual's reported feelings and goals. Consider how past experiences and unconscious patterns might be influencing current states.
4. Use cognitive-behavioral framework to examine the relationship between thoughts, feelings, and behaviors as evidenced in the data and personal report.
5. Integrate holistic health principles to consider how lifestyle factors (sleep, exercise, stress) are impacting overall mental well-being.
6. Synthesize these perspectives to form a cohesive understanding of the individual's current mental health status.
7. Identify areas where more information would be beneficial and explain the psychological rationale behind needing this additional data.

Act as an friendly pyscologist and provide a detailed summary for the step of your reasoning process in 1000 token."""
    )

    second_input_prompt = PromptTemplate(
        input_variables=['description'],
        template="""Given the previous analysis: {description}

As a seasoned psychologist, use advanced chain-of-thought reasoning to:
1. Identify specific mental health symptoms that may be present, referencing both DSM-5 criteria and more nuanced clinical observations.
2. Explain how these symptoms relate to the data and personal information, considering both obvious and subtle connections.
3. Discuss potential underlying causes for these symptoms, drawing from various psychological theories (e.g., attachment theory, stress-diathesis model, cognitive schemas).
4. Analyze how the individual's reported goals and feelings might be manifestations of deeper psychological processes or coping mechanisms.
5. Consider how cultural, social, and environmental factors might be influencing the presentation of symptoms.
6. Highlight any symptoms that require immediate attention and explain why, taking into account both acute risk and long-term impact.
7. Reflect on how your own biases and theoretical orientation might be influencing your interpretation, and consider alternative viewpoints.

Act as an friendly pyscologist and provide a detailed summary for the step of your reasoning process in 1000 token."""
    )

    third_input_prompt = PromptTemplate(
        input_variables=['symptoms'],
        template="""Based on the identified symptoms: {symptoms}

As an experienced psychologist, use advanced chain-of-thought reasoning to:
1. Propose potential interventions or coping strategies for each symptom, drawing from evidence-based practices across multiple therapeutic modalities (e.g., CBT, DBT, ACT, psychodynamic approaches).
2. Explain the rationale behind each intervention, citing relevant psychological research and theories.
3. Discuss how these interventions might interact with the individual's current habits, goals, and physiological data trends.
4. Consider potential contraindications or risks associated with each intervention.
5. Analyze how the proposed interventions might address both surface-level symptoms and underlying psychological processes.
6. Suggest a prioritized action plan for implementing these interventions, taking into account the individual's current resources and readiness for change.
7. Propose methods for integrating the interventions with the individual's existing health routines and goals.

Act as an friendly pyscologist and provide a detailed summary for the step of your reasoning process in 1000 token."""
    )

    fourth_input_prompt = PromptTemplate(
        input_variables=['interventions'],
        template="""Given the proposed interventions: {interventions}

As a psychologist with extensive clinical experience, use advanced chain-of-thought reasoning to:
1. Develop a detailed implementation plan for daily life, considering the individual's unique circumstances and potential barriers.
2. Anticipate potential challenges in implementing these changes and suggest solutions, drawing from motivational interviewing and stages of change models.
3. Identify additional resources or support systems that could aid in this process, including both professional services and community resources.
4. Propose a method for tracking progress and adjusting the plan as needed, incorporating both objective measures and subjective experiences.
5. Discuss how to foster self-efficacy and intrinsic motivation in the individual to sustain long-term change.
6. Consider how to integrate the interventions with the ongoing physiological data collection to create a feedback loop for continuous improvement.
7. Reflect on how to maintain a therapeutic alliance and support the individual through the change process, even in a digital/remote context.

Act as an friendly pyscologist and provide a detailed summary for the step of your reasoning process in 1000 token."""
    )

    # OpenAI LLM with API key
    llm = LangchainOpenAI(temperature=0.8, openai_api_key=openai_api_key)

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
