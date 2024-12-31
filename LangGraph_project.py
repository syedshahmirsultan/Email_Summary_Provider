from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage
from simplegmail import Gmail
from simplegmail.query import construct_query

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App Title
st.title("Email Summary Provider")
st.write("This app will provide you with a summary of your emails. Click the button below to get started.")

# Initialize the LLM with Google API Key
gemini_api = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=gemini_api)

# State Class for LangGraph
class State(MessagesState):
    pass

# System Message for Email Summary
sys_msg = """You are an expert email summary generator. Your task is to summarize the emails I received today in a clear and concise format and don;'t include anykind of Marketing email nd IF I recieved more similiar emials from one sender then only add one of thats summary and at. Use the following example format as a guide:

Today, you received a total of **5 emails**. Here is a summary of your emails:

1. **Subject:** Subject 1  
   **Sender:** Sender 1  
   **Date:** Date 1  
   **Preview:** Preview 1  
   **Message Body:** Message 1  

2. **Subject:** Subject 2  
   **Sender:** Sender 2  
   **Date:** Date 2  
   **Preview:** Preview 2  
   **Message Body:** Message 2  

If there are no emails, respond with:  
" you did not receive any emails today."
"""

# Gmail Function to Fetch and Summarize Emails
def gmailFunc(state: MessagesState):
    gmail = Gmail()
    
    if gmail:
        query_params_1 = {
            "newer_than": (1, "day")
        }
        messages = gmail.get_messages(query=construct_query(query_params_1))
        
        # Collect email details
        total_emails = []
        for message in messages:
            email_details = f"""
Subject: {message.subject}  
Sender: {message.sender}  
Date: {message.date}  
Preview: {message.snippet}
"""
            total_emails.append(email_details)
        
        # Return summarized messages using the LLM
        if total_emails:
            content = sys_msg + "\n".join(total_emails)
            response = llm.invoke([HumanMessage(content=content)])
            return {"messages": [response]}
        else:
            # No emails
            response = llm.invoke([HumanMessage(content=" you did not receive any emails today.")])
            return {"messages": [response]}
    else:
        raise Exception("Gmail not connected")

# Workflow for LangGraph
workflow = StateGraph(MessagesState)

# Adding Nodes and Edges
workflow.add_node('summary_maker', gmailFunc)
workflow.add_edge(START, 'summary_maker')
workflow.add_edge('summary_maker', END)

# Compile the Workflow
graph = workflow.compile()

# Streamlit Button to Trigger Email Summary
click = st.button('Get Your Email Summary')

if click:
    # Invoke the workflow graph to get the email summary
    result = graph.invoke({'messages': []})

    # Display the result in an email-like format
    st.subheader("Email Summary:")
    email_content = f"""
Dear Shahmir,

Here is your email summary for today:

{result['messages'][0].content}

Best regards,  
Your Email Assistant
"""
    st.text(email_content)