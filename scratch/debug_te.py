import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")
from src.voice_nlp_processor import VoiceNLPProcessor

nlp = VoiceNLPProcessor()
text = "తెగిపోయిన విద్యుత్ తీగలపై ఫిర్యాదు."
lang = nlp.detect_language(text)
print("Detected lang:", lang)
print("Civic lexicon result:", nlp._civic_lexicon_translate(text, lang))
print("Full translate result:", nlp.translate_to_english(text, lang))
