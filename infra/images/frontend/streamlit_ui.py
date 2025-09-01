import streamlit as st

# --- Replace with your backend integrations ---
def retrieve_docs(query: str):
    return ["doc1 text", "doc2 text"]

def generate_answer(query: str, docs: list[str]) -> str:
    return f"Answer to '{query}' using {len(docs)} retrieved docs."

# --- Streamlit UI ---
st.set_page_config(page_title="RAG8s Chat", page_icon="💬", layout="wide")
st.title("RAG8s Console")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs = retrieve_docs(prompt)
    answer = generate_answer(prompt, docs)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
