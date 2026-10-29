from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ComplaintDB(Base):
    """SQLAlchemy model for complaints table."""
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(150), nullable=False, index=True)
    sub_category = Column(String(150), nullable=False)
    grievance_date = Column(DateTime, nullable=False)
    ward_name = Column(String(150), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="Registered")
    staff_remarks = Column(Text, nullable=True)
    staff_name = Column(String(100), nullable=True)
    severity_label = Column(String(20), nullable=True)
    recurrence_score = Column(Float, nullable=True, default=0.5)
    priority_score = Column(Float, nullable=True)
    priority_level = Column(String(20), nullable=True)


class WardRiskDB(Base):
    """SQLAlchemy model for wards recurrence risk table."""
    __tablename__ = "wards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ward_name = Column(String(150), unique=True, nullable=False, index=True)
    complaint_count = Column(Integer, default=0)
    category_diversity = Column(Integer, default=1)
    recurrence_score = Column(Float, default=0.5)
    recurrence_level = Column(String(50), default="Medium Recurrence")


class PredictionDB(Base):
    """SQLAlchemy model for user predictions history table."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(150), nullable=False)
    sub_category = Column(String(150), nullable=True)
    ward_name = Column(String(150), nullable=False)
    description = Column(Text, nullable=False)
    predicted_severity = Column(String(20), nullable=False)
    recurrence_risk = Column(String(50), nullable=False)
    priority_score = Column(Float, nullable=False)
    priority_level = Column(String(20), nullable=False)
    explanation = Column(Text, nullable=False)
