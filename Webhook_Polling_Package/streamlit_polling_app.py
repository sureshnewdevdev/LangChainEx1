
import streamlit as st, requests, time

st.title("Polling Demo")
url = "https://jsonplaceholder.typicode.com/posts/1"
slot = st.empty()

while True:
    slot.json(requests.get(url).json())
    time.sleep(3)
