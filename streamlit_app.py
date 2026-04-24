import streamlit as st
import requests

CHAT_API_URL    = "http://app:8000/chat/"
COMPARE_API_URL = "http://app:8000/compare/"

# ── App config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal Advisor Chatbot", page_icon="⚖️", layout="wide")

with st.sidebar:
    st.title("⚖️ Legal Advisor")
    st.caption("Powered by Qdrant + Mistral")
    st.divider()
    page = st.radio(
        "Navigate",
        ["💬 Chat", "🔄 Act Comparison"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Indian Legal Documents\nBNS · CPC · CrPC · Constitution\nCompanies Act · IT Act · MV Act")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ═════════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":
    st.title("💬 Chat")
    st.write("Ask anything about Indian law and get an AI-generated answer with sources.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                    data    = response.json()
                    answer  = data.get("answer", "No response")
                    sources = data.get("sources", [])
                except requests.exceptions.HTTPError:
                    answer  = "⚠️ The AI service is busy. Please wait a moment and try again." \
                              if response.status_code == 429 else f"❌ HTTP Error {response.status_code}"
                    sources = []
                except Exception as e:
                    answer  = f"❌ Error: {str(e)}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ACT COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔄 Act Comparison":
    st.title("🔄 Act Comparison")
    st.write("Enter a topic and two acts — get a **structured side-by-side comparison** fetched directly from the documents.")

    st.info("Examples: `punishment for theft` · `bail provisions` · `right to privacy` · `company incorporation`")

    KNOWN_ACTS = [
        "Bharatiya Nyaya Sanhita", "Indian Penal Code",
        "Code of Civil Procedure", "Code of Criminal Procedure",
        "Constitution of India", "Information Technology Act",
        "Companies Act", "Motor Vehicles Act",
    ]

    with st.form("compare_form"):
        topic = st.text_input("Topic", placeholder="e.g. punishment for theft")
        col1, col2 = st.columns(2)
        with col1:
            act_a = st.selectbox("Act A", KNOWN_ACTS, index=0)
        with col2:
            act_b = st.selectbox("Act B", KNOWN_ACTS, index=1)
        submitted = st.form_submit_button("🔄 Compare Acts")

    if submitted and topic.strip():
        if act_a == act_b:
            st.warning("Please select two different acts.")
        else:
            with st.spinner(f"Comparing '{topic}' across {act_a} and {act_b}…"):
                try:
                    response = requests.post(COMPARE_API_URL,
                                             json={"topic": topic, "act_a": act_a, "act_b": act_b})
                    response.raise_for_status()
                    data       = response.json()
                    comparison = data.get("comparison", "No response")
                    sources_a  = data.get("sources_a", [])
                    sources_b  = data.get("sources_b", [])
                except requests.exceptions.HTTPError:
                    comparison = "⚠️ The AI service is busy. Please wait a moment and try again." \
                                 if response.status_code == 429 else f"❌ HTTP Error {response.status_code}"
                    sources_a = sources_b = []
                except Exception as e:
                    comparison = f"❌ Error: {str(e)}"
                    sources_a = sources_b = []

            st.markdown("### 📊 Comparison Result")
            st.markdown(comparison)
            col1, col2 = st.columns(2)
            with col1:
                if sources_a:
                    with st.expander(f"📚 Sources — {act_a}"):
                        for src in sources_a:
                            st.write(f"- {src}")
            with col2:
                if sources_b:
                    with st.expander(f"📚 Sources — {act_b}"):
                        for src in sources_b:
                            st.write(f"- {src}")

    elif submitted and not topic.strip():
        st.warning("Please enter a topic.")