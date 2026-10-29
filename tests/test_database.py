import pytest
import os
import pandas as pd
from database.database import init_db, SessionLocal, seed_database_from_csv
from database.models import ComplaintDB, WardRiskDB, PredictionDB


def test_database_init_and_seed(tmp_path):
    # Set temp database file
    db_file = tmp_path / "test_civic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    
    init_db()
    db = SessionLocal()
    
    import uuid
    test_id = f"TEST-DB-{str(uuid.uuid4())[:8]}"
    
    # Test record creation
    c = ComplaintDB(
        complaint_id=test_id,
        category="Electrical",
        sub_category="Street Light",
        grievance_date=pd.Timestamp.now(),
        ward_name="Jnanabharathi",
        status="Registered",
        severity_label="HIGH",
        priority_score=75.0,
        priority_level="HIGH"
    )
    db.add(c)
    db.commit()
    
    fetched = db.query(ComplaintDB).filter(ComplaintDB.complaint_id == test_id).first()
    assert fetched is not None
    assert fetched.category == "Electrical"
    assert fetched.priority_score == 75.0
    
    db.close()
