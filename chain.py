## THIS CLASS WILL INITIALIZE THE SUMMARIZER OBJECT

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class TextSummarizer:
    def __init__(self):
        self.model = ChatGroq(
            groq_api_key = os.getenv("GROQ_API_KEY"),
            temperature=0,
            model_name="gemma2-9b-it"
        )

        self.prompt = PromptTemplate.from_template(
                """
                You are a text generator model. 
                You will be given a passage as input
                **PASSAGE TEXT FROM USER:**
                {text}
                **PASSAGE TEXT FROM USER ENDS HERE**

                ** FOLLOWING INFORMATION IS TO BE ANALYSED FROM THE INPUT PASSAGE :- **
                1. Word count: In this field return a numeric value which is the count of number of words present in the passage.
                2. Emotions: In this field return all the emotions present in the passage, like emotion of joy, sadness, anger, etc.
                3. Possible books: In this field return 2-3 possible books from which the input passage text might have been given from.
                4. Summary: In this field return the summary of the passage in 2-3 lines of text.

                ** INSTRUCTIONS: ** Remember to give only the desired output fields, the text will be majorly in english langauge.
                (NO PREAMBLE, ONLY INSTRUCTED OUTPUT REQUIRED and ignore <think> </think> tage in the final output)
                """
            )
        
        self.model_chain = self.prompt | self.model

    
    def get_results(self, text):
        try:
            response = self.model_chain.invoke({"text": text})
            return response.content
    
        except Exception as e:
            return str(e)