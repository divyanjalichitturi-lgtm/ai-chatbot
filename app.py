import streamlit as st
from chatbot import (
    MAX_QUESTION_CHARS,
    ChatbotError,
    ask_bot,
    validate_question,
)

st.set_page_config(
    page_title="College AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College AI Chatbot")
st.write(
    "Ask a question about the college. Answers are generated from the "
    "provided college documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def submit_question(question: str, retry: bool = False) -> None:
    try:
        question = validate_question(question)
    except ValueError as error:
        st.error(str(error))
        return

    if not retry:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the college documents..."):
            try:
                answer = ask_bot(question)
            except (
                ChatbotError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                st.error(str(error))
                st.session_state.last_failed_question = question
            else:
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_failed_question = None


question = st.chat_input(
    "Ask about courses, fees, or the syllabus...",
    max_chars=MAX_QUESTION_CHARS,
)
if question is not None:
    submit_question(question)

last_failed_question = st.session_state.get("last_failed_question")
if last_failed_question and st.button("Retry the last question"):
    submit_question(last_failed_question, retry=True)