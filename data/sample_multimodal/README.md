# Sample Multimodal Complaints Dataset

## Purpose

This directory contains **synthetic sample complaint entries** used for testing the BBMP multimodal complaint pipeline. These records do not require real audio or image files — they serve as structured test fixtures that exercise the full pipeline (language detection, translation, visual inspection, severity classification) without depending on actual citizen-uploaded media.

---

## Directory Structure

```
data/sample_multimodal/
├── README.md               — This file
├── sample_manifest.json    — 10 synthetic complaint entries
├── audio/                  — (optional) Real voice recordings, named <complaint_id>.wav or .mp3
│   └── BBMP-SAMPLE-006.wav (example: Kannada voice note)
└── images/                 — (optional) Real complaint photos, named <complaint_id>.jpg or .png
    └── BBMP-SAMPLE-007.jpg (example: pothole photo)
```

Each entry in `sample_manifest.json` has the following fields:

| Field               | Type        | Description                                                      |
|---------------------|-------------|------------------------------------------------------------------|
| `complaint_id`      | string      | Unique identifier (e.g. `BBMP-SAMPLE-001`)                       |
| `type`              | string      | Input modality: `text`, `voice`, or `image`                      |
| `language`          | string      | ISO 639-1 code: `kn`, `hi`, `ta`, `te`, or `en`                 |
| `text_original`     | string      | Complaint text in the original script/language                   |
| `text_english`      | string      | English translation of the complaint                             |
| `image_description` | string/null | What a real photo would show; `null` for voice/text-only entries |
| `expected_category` | string      | BBMP department category for routing                             |
| `expected_severity` | string      | One of: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`                      |

---

## How to Add Real Audio Files

1. Record or obtain a `.wav` or `.mp3` file of the complaint in the relevant language.
2. Place it at `data/sample_multimodal/audio/<complaint_id>.wav`  
   e.g. `data/sample_multimodal/audio/BBMP-SAMPLE-006.wav`
3. In `sample_manifest.json`, set the entry's `type` field to `"voice"`.
4. The `VoiceNLPProcessor` will automatically pick up and transcribe files at that path.

---

## How to Add Real Images

1. Obtain a `.jpg` or `.png` photograph of the civic issue.
2. Place it at `data/sample_multimodal/images/<complaint_id>.jpg`  
   e.g. `data/sample_multimodal/images/BBMP-SAMPLE-007.jpg`
3. In `sample_manifest.json`, set the entry's `type` field to `"image"` and update `image_description` to describe the photo content.
4. The `VisualInspector` (YOLOv8 / ViT backend) will process the image to measure pothole depth, garbage volume, or fallen tree size.

---

## Dataset References

The following public datasets can be used to populate the `audio/` and `images/` directories with real data:

| Dataset | Source | Use Case |
|---|---|---|
| **ai4bharat/IndicVoices** | [HuggingFace](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Indian language speech data (Kannada, Hindi, Tamil, Telugu) for voice complaints |
| **michelpf/dataset-pothole** | [HuggingFace](https://huggingface.co/datasets/michelpf/dataset-pothole) | Pothole images for visual complaint validation |
| **TACO** | [taco-dataset.github.io](http://taco-dataset.github.io) | Garbage/waste images for overflowing dump detection |
| **ai4bharat/indicnlp-corpus** | [GitHub](https://github.com/ai4bharat/indicnlp_corpus) | Text corpora in Indian languages for NLP model fine-tuning |

---

## Civic Scenarios Covered

The 10 sample entries cover the following five civic complaint scenarios across five languages:

1. **Pothole** — road damage (Kannada text, Hindi image)
2. **Garbage dump** — solid waste overflow (Hindi text, Tamil image)
3. **Streetlight** — broken or non-functional light (Tamil text)
4. **Fallen tree** — blocking road (Telugu text, English image)
5. **Open manhole** — footpath hazard (English text)
6. **Water pipe burst** — Kannada voice
7. **Drainage overflow** — Telugu voice
