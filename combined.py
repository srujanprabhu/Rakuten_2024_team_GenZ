from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import requests
import chainlit as cl
import json
import google.generativeai as genai
from serpapi import GoogleSearch
import fitz  # PyMuPDF
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
import pywhatkit as kit
from IPython.display import Markdown


# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
people_info = {"srujan":["srujanprabhu18@gmail.com", "+917382099233"], "greeshmanth": ["greeshmanthtrade@gmail.com", "+916301918775"], "prabhas":["vanamprabhas899@gmail.com", "+917288990044"], "bhanu":["bhanuprivate2003@gmail.com", "+919701952676"]}

# Initialize the Google LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



base_prompt1 = """
Your name is Racu, an expert and friendly chatbot designed to provide knowledgeable, insightful, and warm assistance across a variety of topics.
You excel in offering detailed guidance, whether it's answering complex questions, providing travel recommendations, or assisting with technical queries.
Your responses should always be clear, concise, and infused with a friendly tone, making interactions pleasant and informative. You adapt your communication style to suit the user's needs, ensuring that each conversation feels personalized and engaging.

Instructions:
- For General Conversations (e.g., What is your name?, what is my name?), just chat friendly with the user, handling the general question using user's query and chat_history, beware you can find user messages as well as your responses in chat history,  feel free to use it.
- For general travel-related queries, offer detailed and enthusiastic advice. Suggest destinations, activities, and accommodations that align with the user's preferences and budget. Inspire users to embark on their dream journeys.(e.g,Plan a trip in a friendly budget).
- If the user asks about real time data (e.g,. "What are the cheapest flight from [location] to [location] on [date]?"  or even "what is the stock price of [stock name]?", "Today's Latest news"), Give query as response, don't add any extra words neither delete!
- If the user asks about IRCTC data,(eg., "what are the cancellation charges?" or "what is vikalp scheme?") then Give response as "irctc", don't add any extra words neither delete!, if you know Answer for user's query for sure, then give the answer.
- If the user asks about PNR status of IRCTC train or flight,(e.g,"What is the status of my ticket with [PNR]?") if user didn't give PNR number then ask the user to give you the PNR Number. Return the response as "irctcPNR,PNRnumber". Strictly!don't add any extra words neither delete.
- If the user asks about Financial related queries(eg.,"What is ITR and how do i file it?" or "Do I need to consult a Charted Accountant for my small business?"), you will become a financial guru and give awesome information.  provide a thorough, informative, consice guidance to the user's query.
- If the user asks to send messages, either in whatsapp or mail(eg.,"Send a message in whatsapp to srujan, send an invitation to Rakathon"). if user wants to send whatsapp message Return the response as "codewhatsapp,name_of_receiver,message", if user want to send a mail, Return the response as "codemail,name_of_receiver,message". Extract these details based on the user query. Don't add extra elements.
- If the user asks about E-commerce products or user asks about an Image that has been uploaded. Return the response as "ecomimage". Strictly!, do not add any other words to this response.

Here’s the conversation so far:
{chat_history}

User Query: {query}
Response:
"""


def get_answer_box(query):
    print("parsed query: ", query)

    search = GoogleSearch({
        "q": query, 
        "api_key": os.getenv('SERP_API_KEY')
    })
    result = search.get_dict()
    
    if 'answer_box' not in result:
        return "No answer box found"
    
    return result['answer_box']

get_answer_box_declaration = {
    'name': "get_answer_box",
    'description': "Get the answer box result for real-time data from a search query",
    'parameters': {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search for"
            }
        },
        "required": [
            "query"
        ]
    },
}



prompt_template1 = PromptTemplate(
    input_variables=["chat_history", "query"],
    template=base_prompt1,
)

# Combine with LLM using LLMChain
llm_chain1 = LLMChain(llm=llm, prompt=prompt_template1)

def send_whatsapp_message(message, whatsapp_number):
    try:
        # Get the current time and add a delay of 2 minutes
        current_time = datetime.now()
        hour = current_time.hour
        minute = current_time.minute + 2

        # Send the WhatsApp message
        kit.sendwhatmsg(whatsapp_number, message, hour, minute)
        return True
    except Exception as e:
        return False
    
def send_mail(message, receiver_mail):
    sender = 'splitit.official@gmail.com'
    password = 'odcy coec gbvf xuqr'
    receiver = receiver_mail

    subject = 'Message from -- LLM'

    body = message
        
    email = EmailMessage()
    email['From'] = sender
    email['To'] = receiver
    email['Subject'] = subject
    email.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context = context) as smtp:
        smtp.login(sender, password=password)
        smtp.sendmail(sender, receiver, email.as_string())
    return True
    
@cl.on_chat_start
async def main():
    cl.user_session.set("chat_history", "")
    cl.user_session.set("llm_chain1", llm_chain1)
    # cl.user_session.set("llm_chain2", llm_chain2)

    initial_message = "Hello! My name is Racu, and I'm your personal Chatbot. How can I help you?"
    cl.user_session.set("chat_history", f"Bot: {initial_message}\n")
    await cl.Message(content=initial_message).send()

@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    chat_history = cl.user_session.get("chat_history", "")  # Retrieve or initialize empty if None

    # Update the chat history with the user's message
    chat_history += f"User: {message.content}\n"
   
    llm_chain1 = cl.user_session.get("llm_chain1")
    # llm_chain2 = cl.user_session.get("llm_chain2")


    user_input = message.content
    inputs = {"chat_history": chat_history, "query": message.content}
    # Generate the response
    response = await llm_chain1.acall(inputs,message.content, callbacks=[cb])
    text_response = response["text"]
    
    def convert_to_dict(bot_response):
        # Remove any extra formatting characters and newlines
        clean_response = bot_response.strip('```json\n{}\n```').replace('\n', '').replace('```', '')

        # Split by commas to get key-value pairs
        items = clean_response.split(',')

        # Initialize an empty dictionary
        response_dict = {}

        # Iterate over items and split by the first colon
        for item in items:
            key, value = item.split(':', 1)

            # Clean up the key and value (remove quotes, strip spaces)
            key = key.strip().strip('"')
            value = value.strip().strip('"')

            # Handle numeric values (convert to float or int)
            if value.replace('.', '', 1).isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            else:
                # Convert to string if not numeric
                value = value

            # Add to dictionary
            response_dict[key] = value

        return response_dict
    
    
    print(user_input)
    print(text_response)
    if user_input.strip() == text_response.strip():
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        # Function to get the answer box
        def get_answer_box(query):
            print("parsed query: ", query)

            search = GoogleSearch({
                "q": query, 
                "api_key": os.getenv('SERP_API_KEY')
            })
            result = search.get_dict()
            
            if 'answer_box' not in result:
                return "No answer box found"
            
            return result['answer_box']

        # Define the function to be used in the model
        get_answer_box_declaration = {
            'name': "get_answer_box",
            'description': "Get the answer box result for real-time data from a search query",
            'parameters': {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for"
                    }
                },
                "required": [
                    "query"
                ]
            },
        }

        # The prompt for the model
        prompt = user_input

        # Generate the response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            tools=[{
                'function_declarations': [get_answer_box_declaration],
            }],
        )

        # Extract function call details from the response
        function_call = response.candidates[0].content.parts[0].function_call
        args = function_call.args
        function_name = function_call.name

        # Initialize result to avoid NameError
        result = None

        # Call the function if the function name matches
        if function_name == 'get_answer_box':
            result = get_answer_box(args['query'])

        # Handle the case where result is not defined or is empty
        if result:
            # Safely truncate and convert to JSON string
            data_from_api = json.dumps(result)[:1500]
        else:
            data_from_api = "No valid data received from get_answer_box."

        # Generate the second response with the data from API
        response2 = model.generate_content(
            f"""
            Based on this information: `{data_from_api}` 
            and this question: `{prompt}`, please respond to the user in a friendly manner, include the links also.
            """,
        )

        print(response2.text)
        chat_history += f"Bot: {response2.text}\n"
        await cl.Message(content=response2.text).send()

    elif text_response.strip() == "irctc":
        import fitz  # PyMuPDF
        query = user_input
        # Open the PDF file
        doc = fitz.open('ircfinal_277.pdf')
        # Extract text from all pages
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()

        base_prompt2 = """
        Answer the user's query based on the given context, the context is a text extracted from pdf files.
        Context:{context}
        User Query :{query}
        """
        llm2 = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
        prompt_template2 = PromptTemplate(
        input_variables=["context", "query"],
        template=base_prompt2,
        )
        llm_chain2 = LLMChain(llm=llm2, prompt=prompt_template2)
        inputs = {"context": text, "query": user_input}
        response_irctc = await llm_chain2.acall(inputs,message.content, callbacks=[cb])
        # Assuming response_irctc is a dictionary or JSON object
        if isinstance(response_irctc, dict):
            response_irctc_text = json.dumps(response_irctc)  # Convert dict to JSON string
        else:
            response_irctc_text = response_irctc  # If it's already a string
        response_irctc_text = response_irctc.get("text", "")
        await cl.Message(content=response_irctc_text).send()

    elif "irctcPNR" in text_response.strip():
        text_response_good = text_response.strip()
        split_result = text_response_good.split(',')
        pnr = split_result.pop()
        url = f"https://irctc-indian-railway-pnr-status.p.rapidapi.com/getPNRStatus/{pnr}"
        headers = {
            "x-rapidapi-key": "7fd5b61ac4msh43ae3074b1df71dp1e507djsn11b5897288c2",
            "x-rapidapi-host": "irctc-indian-railway-pnr-status.p.rapidapi.com"
        }
        response3 = requests.get(url, headers=headers)
        json_data = response3.json()
        base_prompt3 = """
        You are a JSON file reader, You will given a JSON data {json_data}, read that json data and provide insights about that data
        based on the user query:  {query}
        """
        llm3 = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
        prompt_template3 = PromptTemplate(
        input_variables=["json_data", "query"],
        template=base_prompt3,
        )
        llm_chain3 = LLMChain(llm=llm3, prompt=prompt_template3)
        inputs = {"json_data": json_data, "query": user_input}
        response_irctc_pnr = await llm_chain3.acall(inputs,message.content, callbacks=[cb])
        await cl.Message(content=response_irctc_pnr["text"]).send()

    elif "codewhatsapp" in text_response.strip():
        text_response_good = text_response.strip()
        split_result = text_response_good.split(',')
        msg = split_result[2]
        receiver_name = split_result[1]

        base_prompt4 = """
        This is the message that user wants to send - {message}
        You are a message formatter. Your task is to take a brief the input message and expand it into a detailed and well-structured message. 
        Don't ask questions.
        """
        llm4 = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
        prompt_template4 = PromptTemplate(
        input_variables=["message"],
        template=base_prompt4,
        )
        llm_chain4 = LLMChain(llm=llm4, prompt=prompt_template4)
        inputs = {"message":msg}
        response_msg = await llm_chain4.acall(inputs,message.content, callbacks=[cb])
        response_msg = response_msg["text"]
        result = send_whatsapp_message(response_msg, people_info[receiver_name][1])
        if result:
            alert_message = f"message sent sucessfully to {receiver_name} --> {people_info[receiver_name][1]}"
        await cl.Message(content=alert_message).send()
        

    elif "codemail" in text_response.strip():
        text_response_good = text_response.strip()
        split_result = text_response_good.split(',')
        msg = split_result[2]
        receiver_name = split_result[1]
        
        base_prompt5 = """
        This is the message that user wants to send - {message}
        You are a message formatter. Your task is to take a brief the input message and expand it into a detailed and well-structured message. 
        Don't ask questions.
        """
        llm5 = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
        prompt_template5 = PromptTemplate(
        input_variables=["message"],
        template=base_prompt5,
        )
        llm_chain5 = LLMChain(llm=llm5, prompt=prompt_template5)
        inputs = {"message":msg}
        response_msg = await llm_chain5.acall(inputs,message.content, callbacks=[cb])
        response_msg = response_msg["text"]
        result = send_mail(response_msg, people_info[receiver_name][0])
        if result:
            alert_message = f"mail sent sucessfully to {receiver_name} --> {people_info[receiver_name][0]}"
        await cl.Message(content=alert_message).send()

    
    else:
        chat_history += f"Bot: {text_response}\n"

        # Update the chat history in the session
        cl.user_session.set("chat_history", chat_history)
        await cl.Message(content=text_response).send()

    
   