# Import libraries
import datetime
import os
from typing import Any

import boto3
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Bedrock

load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Save to the streamlit memory for chat history
if "memory" not in st.session_state:
    st.session_state.memory = []


class StreamHandler(BaseCallbackHandler):
    """The class is a callback handler that appends new tokens to a text container and displays the updated text in a markdown format.
    """

    def __init__(self, container, initial_text=""):
        """
        The function initializes an object with a container and an initial text value.

        Args:
          container: specify the container or widget where the text
        will be displayed or stored. It could be a text box, label, or any other widget that can display
        text.
          initial_text (optional): set the initial value of the text
        attribute of the object.
        """
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.text += token
        self.container.markdown(self.text)


bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

st.title("Rapyder GenAI Chatbot")


with st.sidebar:
    streaming = st.checkbox("Stream output", value=False,
                            help="This will enable the output of the chatbot to be shown in realtime streaming.")


prompt_template = "You are a helpful assistant chatbot designed to help the humans."

# Chat History
for message in st.session_state.memory:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
st.session_state.memory.append({"role": "system", "content": prompt_template})


prompt = st.chat_input("Ask anything...")

if prompt:
    with st.chat_message("user"):
        st.session_state.memory.append({"role": "Human", "content": prompt})
        st.write(prompt)
        start_time = datetime.datetime.now()
    try:
        with st.chat_message("Assistant"):
            messages = list([st.session_state.memory[0]] +
                            st.session_state.memory[-4:])
            if streaming:
                msg_placeholder = st.empty()
                stream_handler = StreamHandler(msg_placeholder)
                llm = Bedrock(
                    # credentials_profile_name="bedrock-admin",
                    model_id="anthropic.claude-v2",
                    client=bedrock,
                    streaming=True,
                    callbacks=[stream_handler],
                )
                conversation = ConversationChain(
                    llm=llm, verbose=False, memory=ConversationBufferMemory()
                )
                response = conversation.predict(
                    input=prompt)
                end_time = datetime.datetime.now()
                total_time = end_time-start_time
                msg_placeholder.markdown(response)
                st.info(f"Time Taken: {total_time.seconds} seconds")
            else:
                llm = Bedrock(
                    # credentials_profile_name="bedrock-admin",
                    model_id="anthropic.claude-v2",
                    client=bedrock
                )
                conversation = ConversationChain(
                    llm=llm, verbose=False, memory=ConversationBufferMemory()
                )
                response = conversation.predict(input=prompt)
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                st.markdown(response)
                st.info(f"Time taken: {total_time.seconds} seconds")
        st.session_state.memory.append(
            {"role": "Asistant", "content": response})
    except Exception as e:
        st.error(e)
