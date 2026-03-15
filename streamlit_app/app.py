import streamlit as st
import httpx

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Intent Classification Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Intent Classification Chatbot")
st.caption("Powered by BERT — 227 intents | Banking77 + CLINC150 | Redis Memory")

# ── Session State ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "api_status" not in st.session_state:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        st.session_state.api_status = "online" if r.status_code == 200 else "offline"
    except:
        st.session_state.api_status = "offline"

# ── API Status ───────────────────────────────────────────
if st.session_state.api_status == "online":
    st.success("✅ API Connected — Redis Memory Active")
else:
    st.error("❌ API Offline — Start uvicorn on port 8000")
    st.stop()

# ── Chat History ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "intent_data" in msg:
            data = msg["intent_data"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Intent", data["intent"])
            with col2:
                st.metric("Confidence", f"{data['confidence']}%")

            # Low confidence warning
            if data.get("is_low_confidence"):
                st.warning(f"⚠️ Low confidence ({data['confidence']}%) — intent may not be accurate")

            # Previous intent context
            if data.get("last_intent") and data["last_intent"] != data["intent"]:
                st.info(f"💭 Previous intent: {data['last_intent']}")

            with st.expander("Top 3 Predictions"):
                for i, item in enumerate(data["top3"], 1):
                    st.write(f"{i}. **{item['intent']}** — {item['confidence']}%")

# ── Chat Input ───────────────────────────────────────────
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent..."):
            try:
                r = httpx.post(
                    f"{API_URL}/predict",
                    json={
                        "text":       prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=10
                )
                data = r.json()
                st.session_state.session_id = data["session_id"]

                response = f"I detected your intent as **{data['intent']}** with **{data['confidence']}%** confidence."
                st.markdown(response)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Intent", data["intent"])
                with col2:
                    st.metric("Confidence", f"{data['confidence']}%")

                # Low confidence warning
                if data.get("is_low_confidence"):
                    st.warning(f"⚠️ Low confidence ({data['confidence']}%) — intent may not be accurate")

                # Previous intent context
                if data.get("last_intent") and data["last_intent"] != data["intent"]:
                    st.info(f"💭 Previous intent: {data['last_intent']}")

                with st.expander("Top 3 Predictions"):
                    for i, item in enumerate(data["top3"], 1):
                        st.write(f"{i}. **{item['intent']}** — {item['confidence']}%")

                st.session_state.messages.append({
                    "role":        "assistant",
                    "content":     response,
                    "intent_data": data
                })

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": err
                })

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** bert-base-uncased")
    st.write("**Accuracy:** 88.04%")
    st.write("**Intents:** 227")
    st.write("**Memory:** Redis")

    if st.session_state.session_id:
        st.divider()
        st.header("🔑 Session")
        st.code(st.session_state.session_id[:8] + "...")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        if st.session_state.session_id:
            try:
                httpx.delete(f"{API_URL}/history/{st.session_state.session_id}")
            except:
                pass
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()
    st.header("🧪 Sample Queries")
    st.code("Check my account balance")
    st.code("Transfer money to John")
    st.code("My card is not working")
    st.code("I lost my card")