import os
import pytest
from src.voice_nlp_processor import VoiceNLPProcessor
from src.visual_inspector import VisualInspector
from src.multimodal_complaint import MultimodalComplaintProcessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def voice_nlp():
    return VoiceNLPProcessor()


@pytest.fixture
def visual_inspector():
    return VisualInspector()


@pytest.fixture
def multimodal_processor():
    return MultimodalComplaintProcessor()


class TestVoiceNLPProcessor:
    def test_language_detection_indic_scripts(self, voice_nlp):
        # Kannada
        assert voice_nlp.detect_language("ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ") == "kn"
        # Hindi
        assert voice_nlp.detect_language("मुख्य सड़क पर बड़ा गड्ढा है") == "hi"
        # Tamil
        assert voice_nlp.detect_language("தெரு விளக்கு எரியவில்லை") == "ta"
        # Telugu
        assert voice_nlp.detect_language("పెద్ద చెట్టు రోడ్డుపై పడిపోయింది") == "te"
        # English
        assert voice_nlp.detect_language("There is a large pothole on the road") == "en"

    def test_translation_to_english(self, voice_nlp):
        kn_text = "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ"
        translated_kn = voice_nlp.translate_to_english(kn_text, "kn")
        assert any(w in translated_kn.lower() for w in ["pothole", "road", "vehicle", "damage"])

        hi_text = "मुख्य सड़क के पास कचरे का ढेर बहुत बड़ा हो गया है और बदबू आ रही है"
        translated_hi = voice_nlp.translate_to_english(hi_text, "hi")
        assert any(w in translated_hi.lower() for w in ["garbage", "dump", "smell", "waste"])

        en_text = "Street light broken on 4th block"
        translated_en = voice_nlp.translate_to_english(en_text, "en")
        assert translated_en == en_text


class TestVisualInspector:
    def test_inspect_pothole_image(self, visual_inspector):
        img_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-007.jpg")
        if not os.path.exists(img_path):
            pytest.skip("Sample image fixture not found")

        result = visual_inspector.inspect_image(img_path)
        assert result["is_valid_complaint"] is True
        assert result["validation_score"] >= 0.4
        assert "pothole" in result["civic_issues"]
        cat = visual_inspector.map_detection_to_category(result["detections"])
        assert cat == "Roads and Infrastructure"

    def test_inspect_garbage_image(self, visual_inspector):
        img_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-008.jpg")
        if not os.path.exists(img_path):
            pytest.skip("Sample image fixture not found")

        result = visual_inspector.inspect_image(img_path)
        assert result["is_valid_complaint"] is True
        assert "garbage" in result["civic_issues"]
        cat = visual_inspector.map_detection_to_category(result["detections"])
        assert cat == "Solid Waste Management"

    def test_inspect_fallen_tree_image(self, visual_inspector):
        img_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-010.jpg")
        if not os.path.exists(img_path):
            pytest.skip("Sample image fixture not found")

        result = visual_inspector.inspect_image(img_path)
        assert result["is_valid_complaint"] is True
        assert "fallen tree" in result["civic_issues"]
        cat = visual_inspector.map_detection_to_category(result["detections"])
        assert cat == "Tree/Parks Maintenance"

    def test_annotate_image(self, visual_inspector):
        img_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-007.jpg")
        if not os.path.exists(img_path):
            pytest.skip("Sample image fixture not found")

        result = visual_inspector.inspect_image(img_path)
        annotated = visual_inspector.annotate_image(img_path, result["detections"])
        assert annotated is not None
        assert len(annotated.shape) == 3


class TestMultimodalComplaintProcessor:
    def test_process_kannada_complaint(self, multimodal_processor):
        kn_complaint = "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ"
        res = multimodal_processor.process_complaint(text=kn_complaint, ward="Bellandur")

        assert res["detected_language"] == "kn"
        assert res["language_name"] == "Kannada"
        assert len(res["translated_text"]) > 0
        assert res["suggested_category"] == "Roads and Infrastructure"
        assert res["severity_label"] in ["HIGH", "CRITICAL"]
        assert res["priority_score"] >= 60.0
        assert "Roads" in res["department_routed"]

    def test_process_english_critical_manhole(self, multimodal_processor):
        en_complaint = "There is an open manhole on the footpath near Koramangala 5th Block. It is uncovered and poses a serious danger to pedestrians, especially children."
        res = multimodal_processor.process_complaint(text=en_complaint, ward="Koramangala")

        assert res["detected_language"] == "en"
        assert res["severity_label"] == "CRITICAL"
        assert res["priority_level"] in ["HIGH", "CRITICAL"]
        assert res["priority_score"] >= 60.0
        assert "6–12 Hours" in res["target_sla"] or "24–48 Hours" in res["target_sla"]

    def test_process_combined_multimodal(self, multimodal_processor):
        img_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-007.jpg")
        audio_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "audio", "BBMP-SAMPLE-006.wav")

        res = multimodal_processor.process_complaint(
            text="Water pipe burst and road pothole",
            audio_path=audio_path if os.path.exists(audio_path) else None,
            image_path=img_path if os.path.exists(img_path) else None,
            ward="Bellandur",
        )

        assert "complaint_id" in res
        assert "voice_result" in res
        assert "vision_result" in res
        assert "priority_score" in res
        assert 0.0 <= res["priority_score"] <= 100.0
        assert "department_routed" in res
        assert "target_sla" in res
