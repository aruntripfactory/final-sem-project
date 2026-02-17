-- Create research_papers_metadata table
-- PostgreSQL schema for storing research paper metadata

CREATE TABLE research_papers_metadata (
    id SERIAL PRIMARY KEY,
    
    paper_id TEXT UNIQUE,
    title TEXT,
    authors TEXT,
    year INT,
    domain TEXT,
    
    research_problem TEXT,
    objective TEXT,
    methodology TEXT,
    architecture TEXT,
    
    dataset_used TEXT,
    baseline_models TEXT,
    evaluation_metrics TEXT,
    
    key_results TEXT,
    performance_gain TEXT,
    
    key_contributions TEXT,
    limitations TEXT,
    future_work TEXT,
    
    application_area TEXT,
    keywords TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on paper_id for faster lookups
CREATE INDEX idx_research_papers_paper_id ON research_papers_metadata(paper_id);

-- Create index on year for filtering
CREATE INDEX idx_research_papers_year ON research_papers_metadata(year);

-- Create index on domain for filtering
CREATE INDEX idx_research_papers_domain ON research_papers_metadata(domain);

-- Create GIN index on keywords for full-text search (if needed)
-- CREATE INDEX idx_research_papers_keywords_gin ON research_papers_metadata USING gin(to_tsvector('english', keywords));

-- Create GIN index on title for full-text search (if needed)
-- CREATE INDEX idx_research_papers_title_gin ON research_papers_metadata USING gin(to_tsvector('english', title));
