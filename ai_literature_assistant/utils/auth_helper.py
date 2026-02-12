import streamlit as st
import secrets
import logging
from datetime import datetime, timedelta
from database import SessionLocal, User, AuthSession

logger = logging.getLogger(__name__)

def generate_session_token():
    """Generate a secure random session token"""
    return secrets.token_urlsafe(32)

def create_session(user_id, username):
    """Create a new database-backed session and return session token"""
    token = generate_session_token()
    db = SessionLocal()
    try:
        new_session = AuthSession(
            token=token,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        db.add(new_session)
        db.commit()
        logger.info(f"Database session created for user {username}")
        return token
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return None
    finally:
        db.close()

def validate_session(token):
    """Validate session token from database and return user data if valid"""
    if not token:
        return None
    
    db = SessionLocal()
    try:
        session = db.query(AuthSession).filter(AuthSession.token == token).first()
        
        if not session:
            return None
        
        # Check if session expired (24 hours) - matches user's previous logic
        if datetime.utcnow() - session.created_at > timedelta(hours=24):
            logger.info(f"Session expired for token {token[:8]}...")
            db.delete(session)
            db.commit()
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        db.commit()
        
        # Verify user still exists
        user = db.query(User).filter(User.id == session.user_id).first()
        if not user:
            logger.warning(f"User {session.user_id} no longer exists")
            db.delete(session)
            db.commit()
            return None
        
        return {
            'user_id': user.id,
            'username': user.username
        }
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        return None
    finally:
        db.close()

def destroy_session(token):
    """Destroy a session in the database"""
    if not token:
        return
        
    db = SessionLocal()
    try:
        session = db.query(AuthSession).filter(AuthSession.token == token).first()
        if session:
            db.delete(session)
            db.commit()
            logger.info(f"Database session destroyed: {token[:8]}...")
    except Exception as e:
        logger.error(f"Error destroying session: {e}")
    finally:
        db.close()

def init_session_auth():
    """
    Initialize authentication system using query parameters.
    Call this FIRST in your app, right after st.set_page_config()
    """
    # Check query params for token
    query_params = st.query_params.to_dict()
    session_token = query_params.get('session')
    
    if session_token:
        # Validate the session token
        user_data = validate_session(session_token)
        
        if user_data:
            # Valid session - restore user state
            st.session_state.authenticated = True
            st.session_state.user_id = user_data['user_id']
            st.session_state.username = user_data['username']
            st.session_state.session_token = session_token
        else:
            # Invalid/expired session - clear everything
            clear_local_session_state()
            # Remove the invalid token from URL
            if 'session' in st.query_params:
                del st.query_params['session']
            st.rerun()
    
    elif st.session_state.get('authenticated') and st.session_state.get('session_token'):
        # User is authenticated in session_state but URL doesn't have token
        # Add it to URL to persist across page loads
        st.query_params['session'] = st.session_state.session_token

def login_user(user_id, username):
    """Log in a user and set up persistent database session"""
    token = create_session(user_id, username)
    if not token:
        st.error("Failed to create secure session")
        return

    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = username
    st.session_state.session_token = token
    
    # Update query params
    st.query_params['session'] = token
    logger.info(f"User {username} logged in with DB session")

def logout_user():
    """Log out current user and destroy DB session"""
    if 'session_token' in st.session_state:
        destroy_session(st.session_state.session_token)
    
    clear_local_session_state()
    
    # Remove session from URL
    if 'session' in st.query_params:
        del st.query_params['session']
    
    logger.info("User logged out")

def clear_local_session_state():
    """Helper to clear st.session_state auth keys"""
    st.session_state.authenticated = False
    for key in ['user_id', 'username', 'session_token']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear conversation
    st.session_state.conversation = []
    st.session_state.current_session_id = None

def cleanup_expired_sessions():
    """Clean up expired sessions from database"""
    db = SessionLocal()
    try:
        expiry_limit = datetime.utcnow() - timedelta(hours=24)
        expired = db.query(AuthSession).filter(AuthSession.created_at < expiry_limit).all()
        count = len(expired)
        for session in expired:
            db.delete(session)
        db.commit()
        if count > 0:
            logger.info(f"Cleaned up {count} expired database sessions")
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
    finally:
        db.close()