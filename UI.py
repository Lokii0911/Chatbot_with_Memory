import streamlit as st
import uuid
import requests
from requests import session
import os
load_dotenv()
BACKEND_URL = os.environ.get("URL")
st.set_page_config(page_title="Nova AI", page_icon="✨")
st.title("✨ Nova AI Assistant")

if "jwt" not in st.session_state:
    st.session_state.jwt = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "sessions" not in st.session_state:
    st.session_state.sessions = []

if "active_session" not in st.session_state:
    st.session_state.active_session = None

if "messages_map" not in st.session_state:
    st.session_state.messages_map = {}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
def safe_stream_post(url, headers, json):
    res = requests.post(url, headers=headers, json=json, stream=True)

    if res.status_code == 401:
        st.session_state.jwt = None
        st.session_state.user_email = None
        st.session_state.sessions = []
        st.session_state.messages_map = {}
        st.error("Session expired. Please login again.")
        st.stop()

    return res

def safe_request(fn):
    res = fn()
    if res.status_code == 401:
        st.session_state.jwt = None
        st.session_state.user_email = None
        st.session_state.sessions = []
        st.session_state.messages_map = {}
        st.error("Session expired. Please login again.")
        st.stop()
    return res

def auth_ui():
    st.subheader("🔐 Login / Register")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            res = safe_request(
                lambda:requests.post(
                    f"{BACKEND_URL}/auth/login",
                     json={"email": email, "password": password}
                )
            )

            if res.status_code == 200:
                data = res.json()
                st.session_state.jwt = data["access_token"]
                st.session_state.user_email = email
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("Register"):
            res = safe_request(
                lambda:requests.post(
                    f"{BACKEND_URL}/auth/register",
                    json={"email": email, "password": password}
                )
            )

            if res.status_code == 200:
                st.success("Registered! Now login.")
            else:
                st.error("Registration failed")


if not st.session_state.jwt:
    auth_ui()
    st.stop()

def auth_headers():
    return {
        "Authorization": f"Bearer {st.session_state.jwt}"
    }



def load_messages(session_id):
    res = safe_request(
        lambda: requests.get(
            f"{BACKEND_URL}/sessions/{session_id}/messages",
            headers=auth_headers()
        )
    )

    if res.status_code == 200:
        st.session_state.messages_map[session_id] = res.json()
    else:
        st.error("Failed to load messages")


#Syncing memory from backend
def backend_sync():
    try:
        res = safe_request(
            lambda: requests.get(
                f"{BACKEND_URL}/sessions",
                headers=auth_headers()
            )
        )
        if res.status_code != 200:
            st.error("Failed to load sessions")
            return

        st.session_state.sessions = res.json()

        if (
                st.session_state.active_session is None
                and st.session_state.sessions
        ):
            first_session = st.session_state.sessions[0]["id"]
            st.session_state.active_session = first_session

            # Load messages ONLY for active session
            load_messages(first_session)



    except Exception as e:
        st.error(f"Backend error: {e}")


if (
    st.session_state.pending_prompt is None
    and not st.session_state.sessions
):
    backend_sync()

with (st.sidebar):
    st.title("Settings")
    title = st.text_input("Chat title", key="new_chat_title")
    if st.button("New chat", use_container_width=True):
        title = st.session_state.new_chat_title.strip()
        if not title:
            st.warning("Enter a chat title")
        else:
            try:
                response = safe_request(
                    lambda: requests.get(
                        f"{BACKEND_URL}/id",
                        headers=auth_headers(),
                        params={"title": title}
                    )
                )
                if response.status_code == 200:
                    sid = response.json()["session_id"]
                    st.session_state.active_session = sid
                    st.session_state.messages_map[sid] = []
                    backend_sync()
                    st.rerun()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if st.button("Logout"):
        st.session_state.jwt = None
        st.session_state.user_email = None
        st.session_state.sessions = []
        st.session_state.messages_map = {}
        st.rerun()


    st.divider()
    st.header("Chat History")
    for s in st.session_state.sessions:
        sid = s["id"]
        title = s["title"]

        if st.button(title, key=sid, use_container_width=True):
            st.session_state.active_session = sid
            load_messages(sid)
            st.rerun()

    active = st.session_state.active_session

    if not active:
        st.info("Create a new chat 👈")
        st.stop()
active = st.session_state.active_session

if active:
    for msg in st.session_state.messages_map.get(active, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
if active:
    prompt = st.chat_input("Type your message")

    if prompt:
        st.session_state.messages_map[active].append(
            {"role": "user", "content": prompt}
        )
        st.session_state.pending_prompt = prompt

        with st.chat_message("user"):
            st.markdown(prompt)


if st.session_state.pending_prompt:
    with st.chat_message("assistant"):
        spinner_placeholder = st.empty()
        placeholder = st.empty()
        spinner_placeholder.markdown("🤔 **Nova is thinking…**")
        import time
        time.sleep(0.01)

        full_response = ""
        res = safe_stream_post(
            f"{BACKEND_URL}/chat/send",
            headers=auth_headers(),
            json={
                "session_id": active,
                "message": st.session_state.pending_prompt
            }
        )
        with res:
            for line in res.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    token = line.replace("data: ", "")
                    spinner_placeholder.empty()
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
    load_messages(active)

    st.session_state.pending_prompt = None
    st.rerun()






