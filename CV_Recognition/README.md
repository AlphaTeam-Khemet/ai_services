# Egyptian Artifact Recognition API

FastAPI service for artifact classification using the trained **ResNetViTClassifier** (ResNet50 + Vision Transformer) PyTorch model at `model/TransVPR_epoch_1.pt`.

- **Framework:** PyTorch + HuggingFace Transformers
- **Architecture:** ResNet50 feature extractor → ViT encoder → Linear classifier
- **Accuracy:** ~98 % on validation set
- **Classes:** 79 Egyptian landmarks, monuments, and artifacts

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: `http://localhost:8000/docs`

Health check:

```http
GET http://localhost:8000/health
```

Model debug info:

```http
GET http://localhost:8000/debug/model-info
```

Prediction:

```http
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```

Form field:

```text
file: <image file>
```

Example response:

```json
{
  "class_name": "Golden Mask of Tutankhamun",
  "confidence": 0.9832,
  "top_predictions": [
    {"class_name": "Golden Mask of Tutankhamun", "confidence": 0.9832},
    {"class_name": "Golden Throne of Tutankhamun", "confidence": 0.0104},
    {"class_name": "Egyptian Museum, Cairo", "confidence": 0.0031}
  ]
}
```

## Run With Docker

From the project root:

```bash
docker compose up -d cv_recognition
```

> **Note:** `model/TransVPR_epoch_1.pt` must be present before building the image.

## Test the Model Locally

```bash
python test_model.py path/to/your/image.jpg
```

## Classes

The service reads class names from `model/class_names.json` (79 classes):

- Abu Simbel Temples
- Ahmose I
- Akhenaten
- Al-Azhar Mosque
- Al-Deir al-Bahary Temple of Queen Hatshepsut
- Alexandria Library
- Amenhotep III and Tiye
- Amr Ibn al-Aas Mosque
- Bab Zuwayla
- Bab al-Nasr
- Babylon Fortress
- Bagawat
- Baron Empain Palace
- Bayt al-Suhaymi
- Ben Ezra Synagogue
- Bent Pyramid for Senefru
- Cairo Citadel
- Cairo Tower
- Cavern Church Abu Serga
- Cleopatra VII
- Colossoi of Memnon
- Deir el-Medina
- Dendera Temple Complex
- Djoser
- Edfu Temple
- Egyptian Museum, Cairo
- Fatimid Cemetery in Aswan
- Gayer Anderson Museum
- Gebel el-Silsila
- Giza Pyramid Complex
- Goddess Isis with Her Child
- Golden Mask of Tutankhamun
- Golden Throne of Tutankhamun
- Great Hypostyle Hall of Karnak
- Green Head
- Hanging Church (St. Virgin Mary Coptic Orthodox Church)
- Horemheb
- Ibn Tulun Mosque
- Karnak Temple
- Khafre
- Khufu Statue
- King Thutmose III
- Kiosk of Trajan in Philae
- Luxor Temple
- Mausoleum of Aga Khan
- Mortuary Temple of Amenhotep III
- Mummy of Ramsis II
- Narmer (Menes)
- Narmer Palette
- Nefertiti
- Pompeys Pillar Alexandria
- Ptolemaic Temple of Hathor in Deir el-Medina
- Pyramid of Djoser
- Pyramid of Unas
- Qaitbay Citadel
- Queen Hatshepsut
- Ramesseum
- Ramsis II
- Ramsis II Red Granite Statue
- Red Pyramid
- Serapeum of Saqqara
- Sesostris III
- Sobekneferu
- Sphinx
- St. Catherine Monastery Mount Sinai
- St. George Church in Coptic Cairo
- Statue of King Djoser
- Statue of Tutankhamun with Ankhesenamun
- Temple of Habu
- Temple of Hathor
- Temple of Hibis
- Temple of Horus at Edfu
- Temple of Isis in Philae
- Temple of Khonsu in Karnak
- Temple of Kom Ombo
- Temple of Seti I at Abydos
- Temple of the Oracle of Amun at Siwa
- The Solar Boat of Khufu
- The Statue of Sekhmet

## Compare Models

To run the legacy model comparison utility (operates independently of the PyTorch service):

```bash
python compare_models.py \
  "../project data/data/Akhenaten/Akhenaten.jpeg" \
  "../project data/data/Bent_Pyramid_Senefru/Bent_Pyramid_Senefru.jpg"
```
