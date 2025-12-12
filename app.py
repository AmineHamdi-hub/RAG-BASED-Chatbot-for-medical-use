import streamlit as st
from src.llm_wrapper import GroqLLM
from src.agent import MedicalRAGAgent

st.title("🩺 Medical Chatbot")
st.write("This chatbot provides general medical information. **Not a substitute for a doctor.**")

# Initialize Agent
if "agent" not in st.session_state:
    st.session_state.agent = MedicalRAGAgent(GroqLLM())

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

agent = st.session_state.agent

# User input
user_input = st.text_input("Ask a medical question:")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        response = agent.answer(user_input)  # <-- NOW CALL AGENT
    st.session_state.history.insert(0, (user_input, response))

# Display chat history
for user_text, bot_text in st.session_state.history:
    st.markdown(f"**You:** {user_text}")
    st.markdown(f"**Bot:** {bot_text}")
    st.markdown("---")
