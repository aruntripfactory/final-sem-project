#!/usr/bin/env python3
"""Create research_papers_metadata table in PostgreSQL"""

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/ai_literature_assistant")

Base = declarative_base()

class ResearchPaperMetadata(Base):
    __tablename__ = 'research_papers_metadata'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(String(255), unique=True, nullable=False)
    title = Column(Text)
    authors = Column(Text)
    year = Column(Integer)
    domain = Column(Text)
    
    research_problem = Column(Text)
    objective = Column(Text)
    methodology = Column(Text)
    architecture = Column(Text)
    
    dataset_used = Column(Text)
    baseline_models = Column(Text)
    evaluation_metrics = Column(Text)
    
    key_results = Column(Text)
    performance_gain = Column(Text)
    
    key_contributions = Column(Text)
    limitations = Column(Text)
    future_work = Column(Text)
    
    application_area = Column(Text)
    keywords = Column(Text)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

def create_table():
    """Create the research_papers_metadata table"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        print(f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
        
        # Create the table
        Base.metadata.create_all(bind=engine)
        print("✓ Table 'research_papers_metadata' created successfully!")
        
        # Verify table exists
        inspector = inspect(engine)
        if inspector.has_table('research_papers_metadata'):
            columns = inspector.get_columns('research_papers_metadata')
            print(f"\nTable columns ({len(columns)}):")
            for col in columns:
                print(f"  - {col['name']}: {col['type']}")
        else:
            print("✗ Table was not created properly")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating table: {e}")
        return False

if __name__ == "__main__":
    print("Creating research_papers_metadata table...")
    print("-" * 50)
    
    if create_table():
        print("\n✓ Table creation completed successfully!")
    else:
        print("\n✗ Table creation failed!")
        sys.exit(1)
