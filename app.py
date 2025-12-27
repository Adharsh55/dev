from flask import Flask, render_template, request
import pandas as pd 
import google.generativeai as genai
from dotenv import load_dotenv
import os 

#calling API key creating env
load_dotenv()

app=Flask(__name__)

#configure gemini

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-2.5-flash")

#rading the file using the pandas for the framework
df=pd.read_csv("qa_data (1).csv")

#convert csv info to text 
context_text=""
for _, row in df.iterrows():
    context_text +=f"Q:{row['question']}\nA: {row['answer']}\n\n"

def ask_gemini(query):
    promt= f"""
You are a Q&A assistant

Answer only using the context below.
If the answer is not present, say: No relavent Q&A found.

Contxt:
{context_text}

Question: {query}
"""
    response=model.generate_content(promt)
    return response.text.strip()

##Route function
@app.route("/",methods=["GET",'POST'])
def home():
    answer=""
    if request.method=="POST":
        query=request.form["query"]
        answer=ask_gemini(query)
    return render_template("index.html",answer=answer)

##run flask app
if __name__=="__main__":
    app.run()