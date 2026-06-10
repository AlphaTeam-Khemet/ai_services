#!/usr/bin/env python3
"""
Reads all Wikipedia .txt files from the wiki_dataset root directory,
prepends a structured metadata header, and saves each file into
egyptian_rag/data/raw/.
"""

import os
import shutil

# ---------- paths ----------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WIKI_DIR     = os.path.abspath(os.path.join(ROOT_DIR, ".."))   # wiki_dataset/
RAW_DIR      = os.path.join(ROOT_DIR, "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

# ---------- metadata lookup ----------
# Each key is the .txt filename (without extension).
# Values: (topic_id, topic_name, category, period, location)

METADATA = {
    # ── Monuments / Architecture ──────────────────────────────
    "Great_Pyramid_of_Giza": (
        "great_pyramid_of_giza",
        "Great Pyramid of Giza",
        "monument",
        "Old Kingdom, c. 2600 BC",
        "Giza, Egypt",
    ),
    "Great_Sphinx_of_Giza": (
        "great_sphinx_of_giza",
        "Great Sphinx of Giza",
        "monument",
        "Old Kingdom, c. 2500 BC",
        "Giza, Egypt",
    ),
    "Step_Pyramid": (
        "step_pyramid",
        "Step Pyramid of Djoser",
        "monument",
        "Old Kingdom, c. 2670 BC",
        "Saqqara, Egypt",
    ),
    "Red_Pyramid": (
        "red_pyramid",
        "Red Pyramid",
        "monument",
        "Old Kingdom, c. 2590 BC",
        "Dahshur, Egypt",
    ),
    "Bent_Pyramid": (
        "bent_pyramid",
        "Bent Pyramid",
        "monument",
        "Old Kingdom, c. 2600 BC",
        "Dahshur, Egypt",
    ),
    "Pyramids_of_Egypt": (
        "pyramids_of_egypt",
        "Pyramids of Egypt",
        "monument",
        "Old Kingdom – Middle Kingdom",
        "Various locations, Egypt",
    ),
    "Giza_Necropolis": (
        "giza_necropolis",
        "Giza Necropolis",
        "monument",
        "Old Kingdom, c. 2600–2500 BC",
        "Giza, Egypt",
    ),
    "Giza_Plateau": (
        "giza_plateau",
        "Giza Plateau",
        "geography",
        "Old Kingdom onward",
        "Giza, Egypt",
    ),
    "Giza_pyramid_complex": (
        "giza_pyramid_complex",
        "Giza Pyramid Complex",
        "monument",
        "Old Kingdom, c. 2600–2500 BC",
        "Giza, Egypt",
    ),
    "Temple_of_Edfu": (
        "temple_of_edfu",
        "Temple of Edfu",
        "monument",
        "Ptolemaic Period, 237–57 BC",
        "Edfu, Upper Egypt",
    ),
    "Luxor_Temple": (
        "luxor_temple",
        "Luxor Temple",
        "monument",
        "New Kingdom, c. 1400 BC",
        "Luxor (Thebes), Egypt",
    ),
    "Deir_el-Bahari": (
        "deir_el_bahari",
        "Deir el-Bahari",
        "monument",
        "Middle Kingdom – New Kingdom",
        "West Bank, Luxor, Egypt",
    ),
    "Mortuary_Temple_of_Hatshepsut": (
        "mortuary_temple_of_hatshepsut",
        "Mortuary Temple of Hatshepsut",
        "monument",
        "New Kingdom, c. 1479–1458 BC",
        "Deir el-Bahari, Luxor, Egypt",
    ),
    "Colossi_of_Memnon": (
        "colossi_of_memnon",
        "Colossi of Memnon",
        "monument",
        "New Kingdom, c. 1350 BC",
        "West Bank, Luxor, Egypt",
    ),
    "Obelisk": (
        "obelisk",
        "Obelisk",
        "monument",
        "Various periods",
        "Various locations, Egypt",
    ),
    "Saqqara": (
        "saqqara",
        "Saqqara",
        "geography",
        "Early Dynastic Period onward",
        "Saqqara, Memphis, Egypt",
    ),
    "Amarna": (
        "amarna",
        "Amarna (Akhetaten)",
        "geography",
        "New Kingdom, c. 1346–1332 BC",
        "Tell el-Amarna, Middle Egypt",
    ),
    "KV62": (
        "kv62",
        "KV62 – Tomb of Tutankhamun",
        "monument",
        "New Kingdom, c. 1323 BC",
        "Valley of the Kings, Luxor, Egypt",
    ),
    "Tomb_of_Tutankhamun": (
        "tomb_of_tutankhamun",
        "Tomb of Tutankhamun",
        "monument",
        "New Kingdom, c. 1323 BC",
        "Valley of the Kings, Luxor, Egypt",
    ),
    "Grand_Egyptian_Museum": (
        "grand_egyptian_museum",
        "Grand Egyptian Museum",
        "monument",
        "Modern era, opened 2024",
        "Giza, Egypt",
    ),
    "Ancient_Egyptian_architecture": (
        "ancient_egyptian_architecture",
        "Ancient Egyptian Architecture",
        "art",
        "All periods of ancient Egypt",
        "Various locations, Egypt",
    ),

    # ── Pharaohs ──────────────────────────────────────────────
    "Tutankhamun": (
        "tutankhamun",
        "Tutankhamun",
        "pharaoh",
        "New Kingdom, 18th Dynasty, c. 1332–1323 BC",
        "Thebes / Valley of the Kings, Egypt",
    ),
    "Ramses_II": (
        "ramses_ii",
        "Ramesses II (Ramses the Great)",
        "pharaoh",
        "New Kingdom, 19th Dynasty, c. 1279–1213 BC",
        "Pi-Ramesses / Abu Simbel, Egypt",
    ),
    "Hatshepsut": (
        "hatshepsut",
        "Hatshepsut",
        "pharaoh",
        "New Kingdom, 18th Dynasty, c. 1479–1458 BC",
        "Thebes, Egypt",
    ),
    "Thutmose_III": (
        "thutmose_iii",
        "Thutmose III",
        "pharaoh",
        "New Kingdom, 18th Dynasty, c. 1479–1425 BC",
        "Thebes, Egypt",
    ),
    "Amenhotep_III": (
        "amenhotep_iii",
        "Amenhotep III",
        "pharaoh",
        "New Kingdom, 18th Dynasty, c. 1391–1353 BC",
        "Thebes, Egypt",
    ),
    "Narmer": (
        "narmer",
        "Narmer",
        "pharaoh",
        "Early Dynastic Period, c. 3150 BC",
        "Hierakonpolis / Abydos, Egypt",
    ),
    "Sneferu": (
        "sneferu",
        "Sneferu",
        "pharaoh",
        "Old Kingdom, 4th Dynasty, c. 2613–2589 BC",
        "Dahshur / Meidum, Egypt",
    ),
    "Djoser": (
        "djoser",
        "Djoser",
        "pharaoh",
        "Old Kingdom, 3rd Dynasty, c. 2670–2650 BC",
        "Saqqara, Egypt",
    ),
    "Nefertiti": (
        "nefertiti",
        "Nefertiti",
        "pharaoh",
        "New Kingdom, 18th Dynasty, c. 1370–1330 BC",
        "Amarna / Thebes, Egypt",
    ),
    "Seti_I": (
        "seti_i",
        "Seti I",
        "pharaoh",
        "New Kingdom, 19th Dynasty, c. 1294–1279 BC",
        "Thebes / Abydos, Egypt",
    ),
    "Imhotep": (
        "imhotep",
        "Imhotep",
        "historical_figure",  # Vizier, architect & physician — not a pharaoh
        "Old Kingdom, 3rd Dynasty, c. 2650 BC",
        "Saqqara / Memphis, Egypt",
    ),

    # ── Historical Eras ───────────────────────────────────────
    "History_of_ancient_Egypt": (
        "history_of_ancient_egypt",
        "History of Ancient Egypt",
        "historical_era",
        "c. 3150 BC – 30 BC",
        "Egypt",
    ),
    "Old_Kingdom_of_Egypt": (
        "old_kingdom_of_egypt",
        "Old Kingdom of Egypt",
        "historical_era",
        "c. 2686–2181 BC",
        "Memphis, Egypt",
    ),
    "Middle_Kingdom_of_Egypt": (
        "middle_kingdom_of_egypt",
        "Middle Kingdom of Egypt",
        "historical_era",
        "c. 2055–1650 BC",
        "Thebes / Itjtawy, Egypt",
    ),
    "New_Kingdom_of_Egypt": (
        "new_kingdom_of_egypt",
        "New Kingdom of Egypt",
        "historical_era",
        "c. 1550–1077 BC",
        "Thebes, Egypt",
    ),
    "Amarna_Period": (
        "amarna_period",
        "Amarna Period",
        "historical_era",
        "New Kingdom, c. 1353–1336 BC",
        "Amarna, Egypt",
    ),
    "Dynasties_of_Egypt": (
        "dynasties_of_egypt",
        "Dynasties of Ancient Egypt",
        "historical_era",
        "c. 3150 BC – 30 BC",
        "Egypt",
    ),
    "Egyptology": (
        "egyptology",
        "Egyptology",
        "historical_era",
        "Modern era (study of ancient Egypt)",
        "Worldwide",
    ),

    # ── Religion & Mythology ──────────────────────────────────
    "Ancient_Egyptian_religion": (
        "ancient_egyptian_religion",
        "Ancient Egyptian Religion",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Egyptian_mythology": (
        "egyptian_mythology",
        "Egyptian Mythology",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Religion_&_mythology": (
        "religion_and_mythology",
        "Ancient Egyptian Religion & Mythology",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Atenism": (
        "atenism",
        "Atenism",
        "religion",
        "New Kingdom, 18th Dynasty, c. 1353–1336 BC",
        "Amarna, Egypt",
    ),
    "Book_of_the_Dead": (
        "book_of_the_dead",
        "Book of the Dead",
        "religion",
        "New Kingdom onward, c. 1550 BC",
        "Egypt",
    ),
    "Mummification": (
        "mummification",
        "Mummification",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Mummy": (
        "mummy",
        "Mummy",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Ancient_Egyptian_burial_customs": (
        "ancient_egyptian_burial_customs",
        "Ancient Egyptian Burial Customs",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Ma'at": (
        "maat",
        "Ma'at",
        "religion",
        "All periods of ancient Egypt",
        "Egypt",
    ),

    # ── Gods & Goddesses ──────────────────────────────────────
    "Amun": (
        "amun",
        "Amun",
        "religion",
        "Middle Kingdom onward",
        "Thebes (Karnak), Egypt",
    ),
    "Amun-Ra": (
        "amun_ra",
        "Amun-Ra",
        "religion",
        "New Kingdom onward",
        "Thebes (Karnak), Egypt",
    ),
    "Bastet": (
        "bastet",
        "Bastet",
        "religion",
        "Early Dynastic Period onward",
        "Bubastis, Lower Egypt",
    ),
    "Sekhmet": (
        "sekhmet",
        "Sekhmet",
        "religion",
        "Old Kingdom onward",
        "Memphis, Egypt",
    ),
    "Thoth": (
        "thoth",
        "Thoth",
        "religion",
        "Old Kingdom onward",
        "Hermopolis, Egypt",
    ),
    "Hathor": (
        "hathor",
        "Hathor",
        "religion",
        "Old Kingdom onward",
        "Dendera, Egypt",
    ),
    "Ptah": (
        "ptah",
        "Ptah",
        "religion",
        "Early Dynastic Period onward",
        "Memphis, Egypt",
    ),

    # ── Daily Life ────────────────────────────────────────────
    "Ancient_Egyptian_agriculture": (
        "ancient_egyptian_agriculture",
        "Ancient Egyptian Agriculture",
        "daily_life",
        "All periods of ancient Egypt",
        "Nile Valley, Egypt",
    ),
    "Ancient_Egyptian_medicine": (
        "ancient_egyptian_medicine",
        "Ancient Egyptian Medicine",
        "daily_life",
        "All periods of ancient Egypt",
        "Egypt",
    ),

    # ── Art & Literature ──────────────────────────────────────
    "Ancient_Egyptian_art": (
        "ancient_egyptian_art",
        "Ancient Egyptian Art",
        "art",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Ancient_Egyptian_literature": (
        "ancient_egyptian_literature",
        "Ancient Egyptian Literature",
        "art",
        "All periods of ancient Egypt",
        "Egypt",
    ),
    "Hieroglyphics": (
        "hieroglyphics",
        "Egyptian Hieroglyphics",
        "art",
        "Early Dynastic Period onward, c. 3200 BC",
        "Egypt",
    ),

    # ── Artifacts ─────────────────────────────────────────────
    "Tutankhamun's_mask": (
        "tutankhamuns_mask",
        "Tutankhamun's Death Mask",
        "artifact",
        "New Kingdom, 18th Dynasty, c. 1323 BC",
        "Valley of the Kings, Luxor, Egypt",
    ),
    "Rosetta_Stone": (
        "rosetta_stone",
        "Rosetta Stone",
        "artifact",
        "Ptolemaic Period, 196 BC",
        "Rosetta (Rashid), Egypt",
    ),
    "Bust_of_Nefertiti": (
        "bust_of_nefertiti",
        "Bust of Nefertiti",
        "artifact",
        "New Kingdom, 18th Dynasty, c. 1345 BC",
        "Amarna, Egypt",
    ),
    "Narmer_Palette": (
        "narmer_palette",
        "Narmer Palette",
        "artifact",
        "Early Dynastic Period, c. 3100 BC",
        "Hierakonpolis, Egypt",
    ),
    "Khafre_Enthroned": (
        "khafre_enthroned",
        "Khafre Enthroned",
        "artifact",
        "Old Kingdom, 4th Dynasty, c. 2570 BC",
        "Giza, Egypt",
    ),
    "Seated_Scribe": (
        "seated_scribe",
        "Seated Scribe",
        "artifact",
        "Old Kingdom, 4th–5th Dynasty, c. 2620–2500 BC",
        "Saqqara, Egypt",
    ),
    "Sarcophagus": (
        "sarcophagus",
        "Sarcophagus",
        "artifact",
        "All periods of ancient Egypt",
        "Various locations, Egypt",
    ),
    "Shabti": (
        "shabti",
        "Shabti (Ushabti)",
        "artifact",
        "Middle Kingdom onward",
        "Various locations, Egypt",
    ),
    "Cartouche": (
        "cartouche",
        "Cartouche",
        "artifact",
        "Old Kingdom onward",
        "Various locations, Egypt",
    ),
    "Canopic_jar": (
        "canopic_jar",
        "Canopic Jar",
        "artifact",
        "Old Kingdom onward",
        "Various locations, Egypt",
    ),
}


def build_header(topic_id, topic_name, category, period, location):
    """Return the standardized header block."""
    return (
        f"topic_id: {topic_id}\n"
        f"topic_name: {topic_name}\n"
        f"category: {category}\n"
        f"period: {period}\n"
        f"location: {location}\n"
        f"\n---\n\n"
    )


def main():
    source_files = [
        f for f in os.listdir(WIKI_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(WIKI_DIR, f))
    ]
    source_files.sort()

    processed = 0
    skipped = []

    for fname in source_files:
        stem = os.path.splitext(fname)[0]
        if stem not in METADATA:
            skipped.append(fname)
            continue

        topic_id, topic_name, category, period, location = METADATA[stem]
        header = build_header(topic_id, topic_name, category, period, location)

        src_path = os.path.join(WIKI_DIR, fname)
        dst_path = os.path.join(RAW_DIR, fname)

        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(content)

        processed += 1
        print(f"  ✓ {fname}")

    print(f"\n{'='*50}")
    print(f"Processed: {processed} files → egyptian_rag/data/raw/")
    if skipped:
        print(f"Skipped (no metadata mapping): {len(skipped)}")
        for s in skipped:
            print(f"  ✗ {s}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
