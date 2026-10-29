import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from database.models import Base, ComplaintDB, WardRiskDB, PredictionDB

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "civic_grievance.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def init_db():
    """Creates database tables if they do not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DB_PATH}")


def get_db():
    """Dependency session generator for FastAPI and scripts."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_database_from_csv(csv_path: str, ward_csv_path: str = None, limit: int = 5000):
    """
    Seeds database from processed CSV data.
    
    Args:
        csv_path (str): Path to cleaned_grievances.csv.
        ward_csv_path (str, optional): Path to ward_recurrence.csv.
        limit (int): Max records to seed for quick startup.
    """
    init_db()
    db = SessionLocal()
    
    try:
        # Check if already seeded
        if db.query(ComplaintDB).count() > 0:
            print("Database already seeded. Skipping initial seeding.")
            return
            
        print(f"Seeding database from {csv_path} (limit={limit})...")
        df = pd.read_csv(csv_path, low_memory=False)
        if "Grievance Date" in df.columns:
            df["Grievance Date"] = pd.to_datetime(df["Grievance Date"], errors="coerce")
            
        subset = df.head(limit)
        complaint_objs = []
        for _, r in subset.iterrows():
            c_obj = ComplaintDB(
                complaint_id=str(r.get("Complaint ID", "")),
                category=str(r.get("Category_Clean", r.get("Category", ""))),
                sub_category=str(r.get("Sub_Category_Clean", r.get("Sub Category", ""))),
                grievance_date=pd.to_datetime(r.get("Grievance Date", pd.Timestamp.now())),
                ward_name=str(r.get("Ward_Name_Clean", r.get("Ward Name", ""))),
                status=str(r.get("Normalized_Status", r.get("Grievance Status", "Registered"))),
                staff_remarks=str(r.get("Staff Remarks", "")),
                staff_name=str(r.get("Staff Name", "Unassigned")),
                severity_label=str(r.get("severity_label", "MEDIUM")),
                recurrence_score=float(r.get("ward_complaint_density", 0.5)),
                priority_score=float(r.get("priority_score", 50.0)),
                priority_level=str(r.get("priority_level", "MEDIUM"))
            )
            complaint_objs.append(c_obj)
            
        db.bulk_save_objects(complaint_objs)
        db.commit()
        print(f"Successfully seeded {len(complaint_objs):,} complaints to database.")
        
        # Seed Wards
        if ward_csv_path and os.path.exists(ward_csv_path):
            ward_df = pd.read_csv(ward_csv_path)
            ward_objs = []
            for _, wr in ward_df.iterrows():
                w_obj = WardRiskDB(
                    ward_name=str(wr.get("Ward_Name_Clean", "")),
                    complaint_count=int(wr.get("complaint_count", 0)),
                    category_diversity=int(wr.get("category_diversity", 1)),
                    recurrence_score=float(wr.get("recurrence_score", 0.5)),
                    recurrence_level=str(wr.get("recurrence_level", "Medium Recurrence"))
                )
                ward_objs.append(w_obj)
            db.bulk_save_objects(ward_objs)
            db.commit()
            print(f"Successfully seeded {len(ward_objs)} wards to database.")
            
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()
