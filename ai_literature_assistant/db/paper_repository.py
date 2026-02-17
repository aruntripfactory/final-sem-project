from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import ResearchPaperMetadata


def _clean_int(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    value = str(value).strip()

    if value.isdigit():
        return int(value)

    return None


def _clean_text(value):
    """Normalize empty / not specified text."""
    if not value:
        return None

    value = str(value).strip()

    if value.lower() in ["not specified", "n/a", "none", "unknown", ""]:
        return None

    return value


def save_paper_metadata(metadata: dict):

    db: Session = SessionLocal()

    try:
        paper = ResearchPaperMetadata(
            paper_id=metadata.get("paper_id"),
            file_name=_clean_text(metadata.get("file_name")),
            title=_clean_text(metadata.get("title")),
            authors=_clean_text(metadata.get("authors")),
            year=_clean_int(metadata.get("year")),
            domain=_clean_text(metadata.get("domain")),
            research_problem=_clean_text(metadata.get("research_problem")),
            methodology=_clean_text(metadata.get("methodology")),
            dataset=_clean_text(metadata.get("dataset")),
            evaluation_metrics=_clean_text(metadata.get("evaluation_metrics")),
            baseline_models=_clean_text(metadata.get("baseline_models")),
            key_results=_clean_text(metadata.get("key_results")),
            contributions=_clean_text(metadata.get("contributions")),
            limitations=_clean_text(metadata.get("limitations")),
            future_work=_clean_text(metadata.get("future_work")),
            keywords=_clean_text(metadata.get("keywords")),
        )

        db.add(paper)
        db.commit()

    except Exception as e:
        db.rollback()
        raise e

    finally:
        db.close()
