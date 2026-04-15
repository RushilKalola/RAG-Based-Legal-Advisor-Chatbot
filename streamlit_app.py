import streamlit as st
import requests

CHAT_API_URL = "http://app:8000/chat/"
SECTION_API_URL = "http://app:8000/section/"

st.set_page_config(
    page_title="Legal Advisor Chatbot",
    page_icon="⚖️",
    layout="wide"
)

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal Advisor")
    st.caption("Powered by Qdrant + Mistral")
    st.divider()
    page = st.radio("Navigate", ["💬 Chat", "📖 Section Lookup"], label_visibility="collapsed")
    st.divider()
    st.caption("Indian Legal Documents\nBNS · CPC · CrPC · Constitution\nCompanies Act · IT Act · MV Act")


# ─────────────────────────────────────────
# PAGE 1 — CHAT
# ─────────────────────────────────────────
if page == "💬 Chat":
    st.title("💬 Chat")
    st.write("Ask anything about Indian law and get an AI-generated answer with sources.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # chat_input must be at page level — not inside tabs/expander/form
    user_input = st.chat_input("Ask your legal question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(CHAT_API_URL, json={"query": user_input})
                    response.raise_for_status()
                    data = response.json()
                    answer = data.get("answer", "No response")
                    sources = data.get("sources", [])
                except requests.exceptions.HTTPError:
                    if response.status_code == 429:
                        answer = "⚠️ The AI service is busy. Please wait a moment and try again."
                    else:
                        answer = f"❌ HTTP Error {response.status_code}"
                    sources = []
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


# ─────────────────────────────────────────
# PAGE 2 — SECTION LOOKUP
# ─────────────────────────────────────────
elif page == "📖 Section Lookup":
    st.title("📖 Section Lookup")
    st.write("Get the **exact verbatim text** of any section directly from the document.")

    st.info(
        "Examples: `Section 4 Code of Civil Procedure` · `Article 21 Constitution of India` · "
        "`Section 302 Bharatiya Nyaya Sanhita` · `Section 43 Information Technology Act`"
    )

    with st.form("section_form"):
        section_query = st.text_input(
            "Section Reference",
            placeholder="e.g. Section 4 of the Code of Civil Procedure"
        )
        submitted = st.form_submit_button("🔍 Lookup Section")

    if submitted and section_query.strip():
        with st.spinner("Fetching exact section text..."):
            try:
                response = requests.post(SECTION_API_URL, json={"query": section_query})
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No response")
                sources = data.get("sources", [])
            except requests.exceptions.HTTPError:
                if response.status_code == 429:
                    answer = "⚠️ The AI service is busy. Please wait a moment and try again."
                else:
                    answer = f"❌ HTTP Error {response.status_code}"
                sources = []
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                sources = []

        st.markdown("### 📄 Section Text")
        st.markdown(
            f"""<div style="
                background-color: #f8f9fa;
                border-left: 4px solid #1f77b4;
                padding: 16px 20px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                font-size: 0.9rem;
                color: #1a1a1a;
            ">{answer}</div>""",
            unsafe_allow_html=True
        )

        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.write(f"- {src}")

    elif submitted and not section_query.strip():
        st.warning("Please enter a section reference.")