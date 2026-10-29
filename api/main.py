import os
import sys
import datetime
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.severity_model import SeverityClassifier
from src.priority_score import PriorityScorer
from database.database import init_db, get_db, seed_database_from_csv
from database.models import ComplaintDB, WardRiskDB, PredictionDB

# Initialize FastAPI App
app = FastAPI(
    title="Civic Grievance Triage & Resolution-Priority Scoring API",
    description="Operational municipal decision-support REST API for civic complaint prioritization.",
    version="1.0.0"
)

# Global Models Loading
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CLEANED_CSV_PATH = os.path.join(PROCESSED_DIR, "cleaned_grievances.csv")
WARD_CSV_PATH = os.path.join(PROCESSED_DIR, "ward_recurrence.csv")

severity_clf = SeverityClassifier()
try:
    severity_clf.load_model(MODELS_DIR)
except Exception as e:
    print(f"Warning: Could not load severity classifier from {MODELS_DIR}: {e}")

priority_scorer = PriorityScorer()

# Pre-load ward recurrence risk lookup
ward_map = {}
if os.path.exists(WARD_CSV_PATH):
    w_df = pd.read_csv(WARD_CSV_PATH)
    for _, r in w_df.iterrows():
        ward_map[r["Ward_Name_Clean"]] = {
            "recurrence_score": float(r["recurrence_score"]),
            "recurrence_level": str(r["recurrence_level"])
        }

# Initialize Database tables at module import time
init_db()

@app.on_event("startup")
def startup_event():
    """Seeds SQLite database on application startup if needed."""
    if os.path.exists(CLEANED_CSV_PATH):
        seed_database_from_csv(CLEANED_CSV_PATH, WARD_CSV_PATH, limit=5000)

# Request/Response Schemas
class PredictRequest(BaseModel):
    category: str = Field(..., example="Road Maintenance(Engg)")
    sub_category: Optional[str] = Field(default="Potholes", example="Potholes")
    description: str = Field(..., example="Major severe pothole causing accidents near main junction")
    ward: str = Field(..., example="Jnanabharathi Ward")
    complaint_date: Optional[str] = Field(default=None, example="2026-08-20T10:00:00")

class PredictResponse(BaseModel):
    category: str
    ward: str
    severity: str
    recurrence_score: float
    recurrence_level: str
    aging_days: float
    aging_score: float
    priority_score: float
    priority_level: str
    explanation: str

class ComplaintIngestRequest(BaseModel):
    complaint_id: str = Field(..., example="BBMP-2026-99001")
    category: str = Field(..., example="Electrical")
    sub_category: str = Field(..., example="Street Light Not Working")
    ward_name: str = Field(..., example="Horamavu")
    staff_remarks: str = Field(..., example="No streetlight working on 5th cross")
    staff_name: Optional[str] = Field(default="Unassigned")
    grievance_date: Optional[str] = Field(default=None)

# REST Endpoints
@app.get("/", tags=["General"])
def root():
    return {
        "title": "Civic Grievance Triage & Resolution-Priority Scoring System",
        "status": "Operational",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "models_loaded": severity_clf.is_fitted,
        "database_connected": os.path.exists(os.path.join(PROJECT_ROOT, "database", "civic_grievance.db")),
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/statistics", tags=["Analytics"])
def get_statistics(db: Session = Depends(get_db)):
    total = db.query(ComplaintDB).count()
    pending = db.query(ComplaintDB).filter(ComplaintDB.status.in_(["Registered", "In Progress"])).count()
    resolved = db.query(ComplaintDB).filter(ComplaintDB.status.in_(["Closed", "Resolved"])).count()
    critical = db.query(ComplaintDB).filter(ComplaintDB.severity_label == "CRITICAL").count()
    high = db.query(ComplaintDB).filter(ComplaintDB.severity_label == "HIGH").count()
    
    return {
        "total_complaints": total,
        "pending_complaints": pending,
        "resolved_complaints": resolved,
        "critical_severity_count": critical,
        "high_severity_count": high,
        "total_wards": len(ward_map)
    }

@app.get("/complaints", tags=["Complaints"])
def list_complaints(
    ward: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(ComplaintDB)
    if ward:
        query = query.filter(ComplaintDB.ward_name == ward)
    if category:
        query = query.filter(ComplaintDB.category == category)
    if status:
        query = query.filter(ComplaintDB.status == status)
        
    total_count = query.count()
    records = query.offset(offset).limit(limit).all()
    
    return {
        "total_matching": total_count,
        "limit": limit,
        "offset": offset,
        "complaints": [
            {
                "complaint_id": r.complaint_id,
                "category": r.category,
                "sub_category": r.sub_category,
                "ward_name": r.ward_name,
                "grievance_date": r.grievance_date.isoformat() if r.grievance_date else None,
                "status": r.status,
                "severity_label": r.severity_label,
                "priority_score": r.priority_score,
                "priority_level": r.priority_level
            }
            for r in records
        ]
    }

@app.get("/complaints/priority", tags=["Priority Triage"])
def get_priority_worklist(
    priority_level: Optional[str] = None,
    ward: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    db: Session = Depends(get_db)
):
    query = db.query(ComplaintDB)
    if priority_level:
        query = query.filter(ComplaintDB.priority_level == priority_level.upper())
    if ward:
        query = query.filter(ComplaintDB.ward_name == ward)
        
    records = query.order_by(ComplaintDB.priority_score.desc()).limit(limit).all()
    
    worklist = []
    for rank, r in enumerate(records, start=1):
        worklist.append({
            "rank": rank,
            "complaint_id": r.complaint_id,
            "category": r.category,
            "sub_category": r.sub_category,
            "ward_name": r.ward_name,
            "severity_label": r.severity_label,
            "recurrence_score": round((r.recurrence_score or 0.5) * 100, 1),
            "priority_score": r.priority_score,
            "priority_level": r.priority_level
        })
        
    return {"dispatch_ranked_worklist": worklist}

@app.get("/wards/risk", tags=["Ward Risk"])
def get_ward_risk_levels(db: Session = Depends(get_db)):
    wards_db = db.query(WardRiskDB).all()
    if wards_db:
        return [
            {
                "ward_name": w.ward_name,
                "complaint_count": w.complaint_count,
                "category_diversity": w.category_diversity,
                "recurrence_score": w.recurrence_score,
                "recurrence_level": w.recurrence_level
            }
            for w in wards_db
        ]
    # Fallback to in-memory map
    return [
        {"ward_name": k, "recurrence_score": v["recurrence_score"], "recurrence_level": v["recurrence_level"]}
        for k, v in ward_map.items()
    ]

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_triage(req: PredictRequest):
    full_text = f"{req.category} {req.sub_category or ''} {req.description}".lower()
    
    # Severity Prediction
    if severity_clf.is_fitted:
        pred_sev = severity_clf.predict_severity([full_text])[0]
    else:
        from src.feature_engineering import assign_derived_severity
        pred_sev = assign_derived_severity({"Category": req.category, "Sub Category": req.sub_category, "full_text": full_text})
        
    # Recurrence Lookup
    rec_info = ward_map.get(req.ward, {"recurrence_score": 0.5, "recurrence_level": "Medium Recurrence"})
    rec_score = rec_info["recurrence_score"]
    
    # Aging Calculation
    if req.complaint_date:
        try:
            c_date = pd.to_datetime(req.complaint_date)
            aging_days = max(0.0, (pd.Timestamp.now() - c_date).total_seconds() / (24 * 3600))
        except Exception:
            aging_days = 0.0
    else:
        aging_days = 0.0
        
    # Composite Priority Calculation
    res = priority_scorer.calculate_priority_score(pred_sev, rec_score, aging_days)
    
    return PredictResponse(
        category=req.category,
        ward=req.ward,
        severity=pred_sev,
        recurrence_score=round(rec_score * 100.0, 1),
        recurrence_level=rec_info["recurrence_level"],
        aging_days=round(aging_days, 1),
        aging_score=round(res["aging_norm"], 1),
        priority_score=res["priority_score"],
        priority_level=res["priority_level"],
        explanation=res["explanation"]
    )

@app.post("/complaints", tags=["Real-Time Ingestion"])
def ingest_new_complaint(req: ComplaintIngestRequest, db: Session = Depends(get_db)):
    """
    Ingests a new complaint into the system in real time, runs triage pipeline,
    and returns calculated priority score and level.
    """
    full_text = f"{req.category} {req.sub_category} {req.staff_remarks}".lower()
    if severity_clf.is_fitted:
        pred_sev = severity_clf.predict_severity([full_text])[0]
    else:
        from src.feature_engineering import assign_derived_severity
        pred_sev = assign_derived_severity({"Category": req.category, "Sub Category": req.sub_category, "full_text": full_text})
        
    rec_info = ward_map.get(req.ward_name, {"recurrence_score": 0.5, "recurrence_level": "Medium Recurrence"})
    rec_score = rec_info["recurrence_score"]
    
    c_date = pd.to_datetime(req.grievance_date) if req.grievance_date else pd.Timestamp.now()
    aging_days = max(0.0, (pd.Timestamp.now() - c_date).total_seconds() / (24 * 3600))
    
    res = priority_scorer.calculate_priority_score(pred_sev, rec_score, aging_days)
    
    # Save to SQLite Database
    c_obj = ComplaintDB(
        complaint_id=req.complaint_id,
        category=req.category,
        sub_category=req.sub_category,
        grievance_date=c_date,
        ward_name=req.ward_name,
        status="Registered",
        staff_remarks=req.staff_remarks,
        staff_name=req.staff_name or "Unassigned",
        severity_label=pred_sev,
        recurrence_score=rec_score,
        priority_score=res["priority_score"],
        priority_level=res["priority_level"]
    )
    db.add(c_obj)
    db.commit()
    db.refresh(c_obj)
    
    return {
        "message": "Complaint ingested successfully and priority triage score calculated.",
        "complaint_id": req.complaint_id,
        "assigned_severity": pred_sev,
        "recurrence_level": rec_info["recurrence_level"],
        "priority_score": res["priority_score"],
        "priority_level": res["priority_level"],
        "explanation": res["explanation"]
    }

@app.get("/model-performance", tags=["Analytics"])
def get_model_performance():
    metrics_path = os.path.join(MODELS_DIR, "severity_metrics_report.joblib")
    if os.path.exists(metrics_path):
        import joblib
        return joblib.load(metrics_path)
    return {"message": "Metrics report not generated yet."}
