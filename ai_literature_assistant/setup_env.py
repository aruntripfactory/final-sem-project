# ai_literature_assistant/setup_env.py
#!/usr/bin/env python3
"""
Setup environment for AI Research Assistant
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Create .env file if it doesn't exist"""
    
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    
    print("\n" + "="*60)
    print("🔧 SETUP AI RESEARCH ASSISTANT ENVIRONMENT")
    print("="*60)
    
    # Check if .env already exists
    if env_file.exists():
        print(f"\n✅ .env file already exists at: {env_file}")
        print("\nCurrent content:")
        print("-"*40)
        with open(env_file, 'r') as f:
            print(f.read())
        print("-"*40)
        
        response = input("\nDo you want to update it? (y/n): ").lower().strip()
        if response != 'y':
            print("\nKeeping existing .env file.")
            return
    
    # Get API key from user
    print("\n📝 Enter your OpenAI API key (or press Enter to skip):")
    print("   Get it from: https://platform.openai.com/api-keys")
    api_key = input("API Key: ").strip()
    
    # Create .env content
    env_content = f"""# OpenAI API Key
OPENAI_API_KEY={api_key if api_key else 'your-api-key-here'}

# Flask secret key (change this in production!)
FLASK_SECRET_KEY=ai-research-assistant-secret-key-{os.urandom(16).hex()}

# Optional: Other API keys
# ANTHROPIC_API_KEY=
# GOOGLE_API_KEY=
# HUGGINGFACE_TOKEN=
"""
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created .env file at: {env_file}")
    print("\n📋 Content:")
    print("-"*40)
    print(env_content)
    print("-"*40)
    
    # Set file permissions to read-only for user
    try:
        os.chmod(env_file, 0o600)
        print("\n🔐 Set file permissions to read-only for owner")
    except:
        pass
    
    print("\n⚠️  IMPORTANT: Add .env to .gitignore if not already!")
    print("   echo '.env' >> .gitignore")
    
    # Check if config.py exists
    config_file = project_root / "config.py"
    if not config_file.exists():
        print("\n❌ config.py not found! Creating basic config...")
        create_basic_config(project_root)
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Verify your API key in the .env file")
    print("2. Test the system: python query.py --interactive")
    print("3. Ask questions about your research papers!")

def create_basic_config(project_root):
    """Create basic config.py if it doesn't exist"""
    config_content = '''import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
QUERY_RESULTS_DIR = os.path.join(PROJECT_ROOT, "query_results")

# ChromaDB settings
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# OpenAI settings (model only - no API key here!)
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 1500

# RAG settings
MAX_RETRIEVAL_RESULTS = 5
MAX_CONTEXT_LENGTH = 4000
MIN_SIMILARITY_THRESHOLD = 0.3

# Flask settings (non-sensitive)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Create directories if they don't exist
for directory in [DATA_DIR, CHROMA_DIR, PDF_DIR, TEST_RESULTS_DIR, QUERY_RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
'''
    
    config_file = project_root / "config.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created config.py at: {config_file}")

if __name__ == "__main__":
    setup_environment()