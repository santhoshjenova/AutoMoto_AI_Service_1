import streamlit as st
import pandas as pd
from AI_Agent_Service_Reminder import AutoMotoAIServiceReminder
from utils.AI_Agent_Service_Reminder import AutoMotoAIServiceReminder

st.title("AutoMoto AI Service Reminder")

uploaded_file = st.file_uploader("Upload your customer CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    agent = AutoMotoAIServiceReminder(n_clusters=3)
    agent.load_data(df)
    agent.train_model()
    agent.cluster_segments()

    reminders = agent.generate_reminder_table()
    st.write("Service Reminders to Send:", reminders)

    csv = reminders.to_csv(index=False)
    st.download_button("Download reminders CSV", csv, "service_reminders.csv")
