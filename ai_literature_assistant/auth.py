import streamlit as st
import bcrypt
from database import SessionLocal, User, init_db
from utils.auth_helper import login_user, logout_user

# Initialize database
init_db()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

@st.dialog("Authentication Required")
def auth_dialog():
    st.write("Please log in or sign up to upload documents and save your session.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        login_form()
    
    with tab2:
        signup_form()

def login_form():
    if st.session_state.get('signup_success'):
        st.success("Account created! You can now log in.")
        del st.session_state['signup_success']

    with st.form("login_form_dialog"):
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Log In", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please fill in all fields")
                return
                
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.username == username).first()
                if user:
                    if verify_password(password, user.password_hash):
                        # Use the new login system
                        login_user(user.id, user.username)
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Incorrect password")
                else:
                    st.error("User not found")
            finally:
                db.close()

def signup_form():
    with st.form("signup_form_dialog"):
        username = st.text_input("Username", key="signup_user")
        password = st.text_input("Password", type="password", key="signup_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please fill in all fields")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match")
                return
                
            db = SessionLocal()
            try:
                existing_user = db.query(User).filter(User.username == username).first()
                if existing_user:
                    st.error("Username already exists")
                else:
                    hashed = hash_password(password)
                    new_user = User(username=username, password_hash=hashed)
                    db.add(new_user)
                    db.commit()
                    st.session_state.signup_success = True
                    st.rerun()
            finally:
                db.close()

def logout():
    """Logout function to clear session"""
    logout_user()
    st.rerun()

def auth_page():
    """Standalone auth page (for backward compatibility)"""
    st.title("AI Research Literature Assistant")
    choice = st.radio("Access Method", ["Login", "Sign Up"], horizontal=True)
    if choice == "Login":
        login_form()
    else:
        signup_form()
