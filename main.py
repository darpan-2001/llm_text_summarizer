import streamlit as st
from chain import TextSummarizer

st.title("LLM Based Text Summarizer")

## INITIALIZE THE LLM MODEL CHAIN
llm_model = TextSummarizer()

input_passage = st.text_input("Write down the passage here")

if st.button("Analyze"):
    response = llm_model.get_results(input_passage)

    st.subheader("Here is your analysis:-")
    st.write(response)