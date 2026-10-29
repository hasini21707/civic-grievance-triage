import os
import sys

# Ensure current dir is in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

from src.multimodal_complaint import MultimodalComplaintProcessor

proc = MultimodalComplaintProcessor()

tests = [
    ("kn", "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ"),
    ("hi", "मुख्य सड़क के पास कचरे का ढेर बहुत बड़ा हो गया है और बदबू आ रही है"),
    ("ta", "தெரு விளக்கு கடந்த மூன்று நாட்களாக எரியவில்லை, இரவில் நடக்க மிகவும் கஷ்டமாக உள்ளது"),
    ("te", "పెద్ద చెట్టు రోడ్డుపై పడిపోయింది, వాహన రాకపోకలు నిలిచిపోయాయి, వెంటనే తొలగించాలి"),
    ("en", "There is an open manhole on the footpath near Koramangala 5th Block. It is uncovered and poses a serious danger to pedestrians, especially children."),
]

for lang, s in tests:
    res = proc.process_complaint(text=s, ward="Bellandur")
    print("=" * 60)
    print("Original ({}): {}".format(lang, s))
    print("Detected Lang: {} ({})".format(res["detected_language"], res["language_name"]))
    print("Translated:    {}".format(res["translated_text"]))
    print("Category:      {} -> Dept: {}".format(res["suggested_category"], res["department_routed"]))
    print("Severity:      {} | Priority: {} ({})".format(res["severity_label"], res["priority_score"], res["priority_level"]))
    print("SLA Target:    {}".format(res["target_sla"]))
