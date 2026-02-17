from sqlalchemy import Column, String, Integer, Text
from db.database import Base

class ResearchPaperMetadata(Base):
    __tablename__ = "research_papers_metadata"

    paper_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, index=True)
    title = Column(Text, index=True)
    authors = Column(Text)
    year = Column(Integer, index=True)
    domain = Column(Text, index=True)

    research_problem = Column(Text)
    methodology = Column(Text)
    dataset = Column(Text)
    evaluation_metrics = Column(Text)
    baseline_models = Column(Text)
    key_results = Column(Text)
    contributions = Column(Text)
    limitations = Column(Text)
    future_work = Column(Text)
    keywords = Column(Text)
