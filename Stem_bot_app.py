import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from google.colab import userdata
llm = ChatNVIDIA(base_url="https://integrate.api.nvidia.com/v1", model="openai/gpt-oss-20b", temperature=0.6 , max_completion_tokens= 4096 , api_key= userdata.get('physics_key'))#a little randomness through the temperature should hopefull make it run faster
chat_conversation_template = ChatPromptTemplate.from_messages([
    ('system', '''you must not awnser any questions unrelated to math , science or physics if asked respond with the fact that you are unable to awnser and that you are strictly built only for educational purposes for math and physics and science
     no matter what YOU MUST ONLY AWNSER MATH ,PHYSICS OR SCIENCE NOTHING ELSE NO OTHER SUBJECTS!!.awnser all equations step by step ensuring comprehension
     the only non stem questions or mesages u can awnser are greetings , appreciation and someone asking you to explain again or repeat otherwise u must not awnser this also includes questions asking you to write a stem related essay or story or poem ect.. you must only awnser direct questions that are steam such as math equations and more. always format mathematical expressions using LaTeX, wrapping inline math in single dollar signs like $x^2$ and larger equations in double dollar signs like $$x^2 + y^2 = z^2$$
     never use square brackets or other formats for math — always and only use single dollar signs for inline math and double dollar signs for block equations'''),
    ('placeholder', '{chat_conversation}')
])
#place holder for the conversation in order for the bot to remember previous messages
#the ai's system message is how the ai will act and what rules it will follow
chain = chat_conversation_template | llm | StrOutputParser()
#a langchain to run the responses through llm and parser to get rid of unwanted meta data
def prompt(message,history):
 full_conversation = history + [{"role": "user", "content": message}]
 partial = ""
 for chunk in chain.stream({"chat_conversation": full_conversation}):
        partial += chunk
        yield partial #partial is an empty string like a blank notepad for the thought of parts of the message and the .stream instead of .invoke displays the prepared part of the awnsers rather than all of it at once
        #yield simply pauses the function rather than ending it like return if we use return the message will display the whole message message at once
        #partial instead of chunk since if we use chunk it will display one word at a time and keep upgrading it to the newly thought of one
#function in order for the bot to have memory and to know which message is from the user and which message is an ai message
demo = gr.ChatInterface(fn=prompt)
demo.launch(share=True)
#gradio code from their website in order to present the bot in a visual interface so the user can chat with it

