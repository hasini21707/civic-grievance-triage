"""
MultimodalComplaintProcessor — Unified pipeline for multimodal BBMP citizen complaints.

Orchestrates VoiceNLPProcessor, VisualInspector, SeverityClassifier, and PriorityScorer
to process complaints submitted via text, voice, or images.
"""

import os
import uuid
import datetime
from typing import Optional, Dict, Any

from src.voice_nlp_processor import VoiceNLPProcessor, LANGUAGE_MAP
from src.visual_inspector import VisualInspector
from src.severity_model import SeverityClassifier
from src.priority_score import PriorityScorer
from src.feature_engineering import CRITICAL_KEYWORDS, HIGH_KEYWORDS, MEDIUM_KEYWORDS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BBMP category keywords for text-based domain inference
_CATEGORY_KEYWORDS = {
    "Roads and Infrastructure": [
        "pothole", "road", "pavement", "manhole", "footpath", "bridge",
        "road cave", "road damage", "asphalt", "crater", "cracked road", "traffic",
        "ಹೊಂಡ", "ಗುಂಡಿ", "ರಸ್ತೆ", "ಗಡ್ಡಾ", "गड्ढा", "सड़क", "குழி", "சாலை", "గుంత", "రోడ్డు"
    ],
    "Solid Waste Management": [
        "garbage", "waste", "trash", "dump", "litter", "debris", "sweeping",
        "dumpster", "rubbish", "stench", "foul smell",
        "ಕಸ", "ತ್ಯಾಜ್ಯ", "ಕಸದ ರಾಶಿ", "कचरा", "बदबू", "குப்பை", "చెత్త"
    ],
    "Streetlights": [
        "streetlight", "street light", "lamp post", "light pole", "dark",
        "power line", "power lines", "wire", "cable", "wires", "downed wire",
        "electric", "electrical", "live wire", "snapped wire", "transformer", "sparking",
        "బీది ದೀಪ", "ದೀಪ", "ವಿದ್ಯುತ್", "ತಂತಿ", "ತಂತಿಗಳು", "स्ट्रीट लाइट", "बत्ती", "बिजली", "तार",
        "விளக்கு", "மின்சாரம்", "கம்பி", "లైటు", "విద్యుత్", "తీగలు", "తీగలపై", "తీగ"
    ],
    "Water Supply": [
        "water", "leakage", "pipe", "burst pipe", "drinking water", "supply",
        "ನೀರು", "ಪೈಪ್", "पानी", "पाइप", "தண்ணீர்", "குழாய்", "పైపు"
    ],
    "Tree/Parks Maintenance": [
        "tree", "branch", "park", "fallen tree", "dead tree", "uprooted",
        "ಮರ", "पेड़", "மரம்", "చెట్టు"
    ],
    "Storm Water Drain": [
        "flooding", "flood", "drain", "overflow", "stagnant water", "swd",
        "storm water", "sewage", "sewer",
        "ಚರಂಡಿ", "ಮುಳುಗಡೆ", "नाली", "வடிகால்", "சாக்கடை", "మురుగు"
    ],
    "Animal Menace": [
        "dog", "stray", "animal", "cat", "monkey", "cattle", "dog bite",
        "ನಾಯಿ", "कुत्ता", "நாய்", "కుక్క"
    ],
}

# Municipal department routing directory
DEPARTMENT_ROUTING = {
    "Roads and Infrastructure": "BBMP Major Roads & Infrastructure Division",
    "Solid Waste Management": "BBMP Solid Waste Management (SWM) Cell",
    "Streetlights": "BESCOM & BBMP Electrical Infrastructure Wing",
    "Water Supply": "BWSSB Water Supply & Sewerage Board",
    "Tree/Parks Maintenance": "BBMP Forest Cell & Horticulture Department",
    "Storm Water Drain": "BBMP Storm Water Drain (SWD) Department",
    "Animal Menace": "BBMP Animal Husbandry & Veterinary Department",
    "General": "BBMP Central Grievance Redressal Cell",
}

# Resolution SLA recommendations based on priority level
SLA_TARGETS = {
    "CRITICAL": "Resolution within 6–12 Hours (Emergency Dispatch)",
    "HIGH": "Resolution within 24–48 Hours",
    "MEDIUM": "Resolution within 3–5 Business Days",
    "LOW": "Resolution within 7 Business Days (Scheduled Maintenance)",
}


class MultimodalComplaintProcessor:
    """
    End-to-end multimodal complaint processor.
    Accepts text, audio path, and/or image path; returns a unified
    structured complaint record with severity, priority, and validation metadata.
    """

    def __init__(self):
        self.voice_processor = VoiceNLPProcessor()
        self.visual_inspector = VisualInspector()
        self.scorer = PriorityScorer()

        # Attempt to load trained SeverityClassifier; fall back to rule-based
        self._classifier: Optional[SeverityClassifier] = None
        models_dir = os.path.join(PROJECT_ROOT, "models")
        try:
            clf = SeverityClassifier()
            clf.load_model(models_dir)
            self._classifier = clf
        except Exception:
            self._classifier = None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def generate_complaint_id(self) -> str:
        """Return a unique complaint ID string."""
        return f"BBMP-MM-{uuid.uuid4().hex[:8].upper()}"

    def validate_complaint(self, result: dict) -> bool:
        """
        Cross-modal contradiction check.
        Returns False when visual severity is CRITICAL but text indicates low/routine matter,
        or vice-versa with significant contradiction.
        """
        visual_severity = result.get("visual_severity", "")
        text_severity = result.get("severity_label", "")
        if visual_severity == "CRITICAL" and text_severity == "LOW":
            return False
        return True

    def _rule_based_severity(self, text: str) -> str:
        """
        Keyword-based severity estimator using standard BBMP civic risk factors.
        """
        text_lower = text.lower()
        for kw in CRITICAL_KEYWORDS:
            if kw in text_lower:
                return "CRITICAL"
        for kw in HIGH_KEYWORDS:
            if kw in text_lower:
                return "HIGH"
        for kw in MEDIUM_KEYWORDS:
            if kw in text_lower:
                return "MEDIUM"
        return "LOW"

    def _predict_hybrid_severity(self, processed_text: str, visual_severity: str = "LOW") -> str:
        """
        Hybrid severity estimator: guarantees that life-safety hazards identified
        by domain indicators are never downgraded, while utilizing ML classifier
        for nuances and fusing visual evidence.
        """
        rule_sev = self._rule_based_severity(processed_text) if processed_text else "LOW"

        ml_sev = "LOW"
        if self._classifier is not None and processed_text:
            try:
                ml_sev = self._classifier.predict_severity([processed_text])[0]
            except Exception:
                ml_sev = rule_sev

        # Hierarchy: CRITICAL > HIGH > MEDIUM > LOW
        ranks = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        # If rule explicitly identified a critical or high hazard, trust domain safety rule
        if ranks.get(rule_sev, 1) >= 3:
            base_sev = rule_sev
        else:
            base_sev = ml_sev if ranks.get(ml_sev, 1) >= ranks.get(rule_sev, 1) else rule_sev

        # If visual inspection verified a critical hazard (e.g. huge cavity or fallen tree)
        if ranks.get(visual_severity, 1) > ranks.get(base_sev, 1):
            # Promote by one tier
            inv_ranks = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}
            promoted_rank = min(4, ranks.get(base_sev, 1) + 1)
            return inv_ranks[promoted_rank]

        return base_sev

    def _infer_category_from_text(self, text: str) -> str:
        """Guess the BBMP complaint category from text keywords."""
        if not text:
            return "General"
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for category, keywords in _CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score:
                scores[category] = score
        if scores:
            return max(scores, key=scores.__getitem__)
        return "General"

    def _lookup_recurrence_score(self, ward: str) -> float:
        """
        Look up ward recurrence risk score. Defaults to 0.5 if not found.
        """
        recurrence_csv = os.path.join(PROJECT_ROOT, "data", "processed", "ward_recurrence.csv")
        if not os.path.exists(recurrence_csv):
            return 0.5

        try:
            import pandas as pd
            df = pd.read_csv(recurrence_csv)
            df.columns = [c.strip() for c in df.columns]

            ward_col = next((c for c in df.columns if "ward" in c.lower()), None)
            score_col = next((c for c in df.columns if "recurrence" in c.lower() or "score" in c.lower()), None)

            if not ward_col or not score_col:
                return 0.5

            clean_ward = ward.lower().strip()
            match = df[df[ward_col].astype(str).str.lower().str.contains(clean_ward, regex=False)]
            if match.empty:
                return 0.5

            return float(match.iloc[0][score_col])
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_complaint(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        ward: Optional[str] = None,
        complaint_date: Any = None,
    ) -> dict:
        """
        Process a multimodal complaint submission end-to-end.

        Args:
            text:           Raw complaint text (in English or regional language).
            audio_path:     Path to audio file for Whisper transcription.
            image_path:     Path to image or video file for visual inspection.
            ward:           Ward name for recurrence lookup.
            complaint_date: datetime or date string for aging calculation.

        Returns:
            Structured dictionary with complete triage, intermediate analysis,
            SLA, routing, and verification metrics.
        """
        complaint_id = self.generate_complaint_id()
        original_inputs = {
            "text": text,
            "audio_path": audio_path,
            "image_path": image_path,
            "ward": ward,
        }

        # ------------------------------------------------------------------
        # 1. Voice processing & Language translation
        # ------------------------------------------------------------------
        voice_result = None
        raw_input_text = text.strip() if text else ""
        detected_language = "en"
        translated_text = ""

        if audio_path and os.path.exists(audio_path):
            voice_result = self.voice_processor.process_voice_complaint(audio_path)
            if not voice_result.get("requires_installation"):
                audio_text = voice_result.get("original_text", "")
                if audio_text:
                    detected_language = voice_result.get("detected_language", "en")
                    translated_text = voice_result.get("translated_text", audio_text)
                    if not raw_input_text:
                        raw_input_text = audio_text

        # ------------------------------------------------------------------
        # 2. Text Language Detection & Translation (if text provided directly)
        # ------------------------------------------------------------------
        if raw_input_text and not translated_text:
            detected_language = self.voice_processor.detect_language(raw_input_text)
            if detected_language != "en":
                translated_text = self.voice_processor.translate_to_english(
                    raw_input_text, detected_language
                )
            else:
                translated_text = raw_input_text

        # The processed English text used for downstream NLP
        processed_text = translated_text if translated_text else raw_input_text

        # Structure voice/NLP intermediate analysis for dashboard
        nlp_intermediate = {
            "detected_language": detected_language,
            "language_name": LANGUAGE_MAP.get(detected_language, detected_language.upper()),
            "original_text": raw_input_text,
            "translated_text": translated_text if detected_language != "en" else "",
            "confidence": voice_result.get("confidence", 0.95) if voice_result else 0.98,
        }

        # ------------------------------------------------------------------
        # 3. Visual inspection
        # ------------------------------------------------------------------
        visual_result = None
        visual_detections = []
        visual_severity = "LOW"
        visual_validation_score = 0.0

        if image_path and os.path.exists(image_path):
            visual_result = self.visual_inspector.inspect_image(image_path)
            visual_detections = visual_result.get("detections", [])
            visual_severity = visual_result.get("estimated_severity", "LOW")
            visual_validation_score = visual_result.get("validation_score", 0.0)

        # ------------------------------------------------------------------
        # 4. Severity prediction (Hybrid safety + ML + vision)
        # ------------------------------------------------------------------
        severity_label = self._predict_hybrid_severity(processed_text, visual_severity)

        # ------------------------------------------------------------------
        # 5. Recurrence score
        # ------------------------------------------------------------------
        recurrence_score = 0.5
        if ward:
            recurrence_score = self._lookup_recurrence_score(ward)

        # ------------------------------------------------------------------
        # 6. Days pending
        # ------------------------------------------------------------------
        days_pending = 0.0
        if complaint_date is not None:
            try:
                if isinstance(complaint_date, datetime.datetime):
                    delta = datetime.datetime.now() - complaint_date
                    days_pending = max(0.0, delta.total_seconds() / 86400.0)
                elif isinstance(complaint_date, datetime.date):
                    delta = datetime.date.today() - complaint_date
                    days_pending = max(0.0, float(delta.days))
                elif isinstance(complaint_date, str):
                    dt = datetime.datetime.fromisoformat(complaint_date.replace("Z", ""))
                    delta = datetime.datetime.now() - dt
                    days_pending = max(0.0, delta.total_seconds() / 86400.0)
            except Exception:
                days_pending = 0.0

        # ------------------------------------------------------------------
        # 7. Priority scoring
        # ------------------------------------------------------------------
        priority_result = self.scorer.calculate_priority_score(
            severity_label, recurrence_score, days_pending
        )

        # ------------------------------------------------------------------
        # 8. Category suggestion & Department routing
        # ------------------------------------------------------------------
        if visual_result and visual_detections and visual_validation_score >= 0.5:
            suggested_category = self.visual_inspector.map_detection_to_category(visual_detections)
        else:
            suggested_category = self._infer_category_from_text(processed_text)
            if suggested_category == "General" and visual_detections:
                suggested_category = self.visual_inspector.map_detection_to_category(visual_detections)

        department_routed = DEPARTMENT_ROUTING.get(
            suggested_category, "BBMP Central Grievance Redressal Cell"
        )
        target_sla = SLA_TARGETS.get(
            priority_result["priority_level"], "Resolution within 3–5 Business Days"
        )

        # ------------------------------------------------------------------
        # 9. Validation & Human review flags
        # ------------------------------------------------------------------
        intermediate_check = {
            "visual_severity": visual_severity,
            "severity_label": severity_label,
        }
        is_valid = self.validate_complaint(intermediate_check)
        requires_human_review = (
            (not is_valid)
            or (image_path is not None and visual_validation_score < 0.35)
            or (not processed_text and not visual_detections)
        )

        # ------------------------------------------------------------------
        # 10. Assemble complete result
        # ------------------------------------------------------------------
        return {
            "complaint_id": complaint_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "original_inputs": original_inputs,
            "original_text": raw_input_text,
            "translated_text": translated_text if detected_language != "en" else "",
            "processed_text": processed_text,
            "detected_language": detected_language,
            "language_name": LANGUAGE_MAP.get(detected_language, "English"),
            "voice_result": voice_result or nlp_intermediate,
            "vision_result": visual_result,
            "visual_detections": visual_detections,
            "visual_severity": visual_severity,
            "suggested_category": suggested_category,
            "department_routed": department_routed,
            "target_sla": target_sla,
            "severity_label": severity_label,
            "recurrence_score": recurrence_score,
            "days_pending": round(days_pending, 2),
            "priority_score": priority_result["priority_score"],
            "priority_level": priority_result["priority_level"],
            "validation_score": visual_validation_score,
            "requires_human_review": requires_human_review,
            "explanation": priority_result["explanation"],
        }
