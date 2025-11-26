#!/usr/bin/env python3
"""
Test script for Conversational RAG System (LM Studio + Snowflake embeddings).

- Runs 10 English and 10 Turkish queries (same intents).
- Queries cover:
  * Sasa Polyester activity reports (single-year document search)
  * ISO 2021-2024 lists (single-year SQL/data queries, matching the ISO table schema)
  * Email drafting
  * General chat
- Repeats the full query set for N runs (default: 5).
- Writes all queries and responses into a text file under data/logs/.
- Output file name is given as a command-line argument (without extension).
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# IMPORTANT: adjust this import to the actual module name where
# create_console_rag_system is defined (the big script you pasted).
from conversation import create_console_rag_system  # type: ignore


def build_test_queries() -> List[Dict[str, Any]]:
    """Build 10 English + 10 Turkish queries with matching intents.

    All Sasa and ISO questions are single-year, document-specific or table-specific.
    ISO questions are written according to the ISO table schema.
    """

    queries: List[Dict[str, Any]] = []

    # ---------------- Sasa Polyester (faaliyet raporu, tek yıl) ----------------

    en_sasa = [
        {
            "id": "sasa_2022_en",
            "lang": "EN",
            "category": "SASA_DOC_2022",
            "text": (
                "In Sasa Polyester Sanayi A.Ş. 2022 annual report, "
                "what are the main strategic priorities stated by management?"
            ),
        },
        {
            "id": "sasa_2023_en",
            "lang": "EN",
            "category": "SASA_DOC_2023",
            "text": (
                "In Sasa Polyester Sanayi A.Ş. 2023 annual report, "
                "summarize the key investment projects planned for the next year."
            ),
        },
        {
            "id": "sasa_2024_en",
            "lang": "EN",
            "category": "SASA_DOC_2024",
            "text": (
                "In Sasa Polyester Sanayi A.Ş. 2024 annual report, "
                "what risks related to raw material prices are highlighted?"
            ),
        },
    ]

    tr_sasa = [
        {
            "id": "sasa_2022_tr",
            "lang": "TR",
            "category": "SASA_DOC_2022",
            "text": (
                "Sasa Polyester Sanayi A.Ş.'nin 2022 faaliyet raporunda, "
                "yönetimin belirttiği temel stratejik öncelikler nelerdir?"
            ),
        },
        {
            "id": "sasa_2023_tr",
            "lang": "TR",
            "category": "SASA_DOC_2023",
            "text": (
                "Sasa Polyester Sanayi A.Ş.'nin 2023 faaliyet raporunda, "
                "bir sonraki yıl için planlanan başlıca yatırım projelerini özetler misin?"
            ),
        },
        {
            "id": "sasa_2024_tr",
            "lang": "TR",
            "category": "SASA_DOC_2024",
            "text": (
                "Sasa Polyester Sanayi A.Ş.'nin 2024 faaliyet raporunda, "
                "hammadde fiyatlarıyla ilgili vurgulanan başlıca riskler nelerdir?"
            ),
        },
    ]

    # ---------------- ISO listesi (SQL, tablo şemasına uygun, tek yıl) ---------

    # ISO tablosu için önemli kolonlar:
    # - Kuruluş Adı
    # - Yıl
    # - Üretimden Satışlar (Net) (TL)
    # - Net Satışlar (TL)
    # - FAVÖK (TL)
    # - İhracat (Bin $)
    # - Ücretle Çalışanlar Ortalaması (Kişi)
    # - NACE Kodu
    # - ISIC Sektör Tanımı

    en_iso = [
        {
            "id": "iso_2021_en",
            "lang": "EN",
            "category": "ISO_SQL_2021",
            "text": (
                "In the ISO 2021 list table, find the company with the highest value in "
                "the column 'Üretimden Satışlar (Net) (TL)' and return its "
                "'Kuruluş Adı', 'Genel Sıra No', and 'Yıl'."
            ),
        },
        {
            "id": "iso_2022_en",
            "lang": "EN",
            "category": "ISO_SQL_2022",
            "text": (
                "Using the ISO 2022 list table, list the top 10 companies ordered by "
                "'İhracat (Bin $)'. For each company, return 'Kuruluş Adı', "
                "'İhracat (Bin $)', and 'NACE Kodu'."
            ),
        },
        {
            "id": "iso_2023_en",
            "lang": "EN",
            "category": "ISO_SQL_2023",
            "text": (
                "From the ISO 2023 list table, calculate the total "
                "'Ücretle Çalışanlar Ortalaması (Kişi)' for all companies with 'NACE Kodu' = 29."
            ),
        },
        {
            "id": "iso_2024_en",
            "lang": "EN",
            "category": "ISO_SQL_2024",
            "text": (
                "In the ISO 2024 list table, for companies where 'ISIC Sektör Tanımı' is "
                "'Taşıt Araçları Sanayi', compute the average 'FAVÖK (TL)'."
            ),
        },
    ]

    tr_iso = [
        {
            "id": "iso_2021_tr",
            "lang": "TR",
            "category": "ISO_SQL_2021",
            "text": (
                "ISO 2021 listesi tablosunda 'Üretimden Satışlar (Net) (TL)' kolonu "
                "en yüksek olan şirketi bul ve bu şirketin 'Kuruluş Adı', "
                "'Genel Sıra No' ve 'Yıl' bilgilerini döndür."
            ),
        },
        {
            "id": "iso_2022_tr",
            "lang": "TR",
            "category": "ISO_SQL_2022",
            "text": (
                "ISO 2022 listesi tablosunu kullanarak şirketleri 'İhracat (Bin $)' "
                "kolonuna göre azalan sırada listele ve ilk 10 şirketi getir. "
                "Her şirket için 'Kuruluş Adı', 'İhracat (Bin $)' ve 'NACE Kodu' "
                "alanlarını döndür."
            ),
        },
        {
            "id": "iso_2023_tr",
            "lang": "TR",
            "category": "ISO_SQL_2023",
            "text": (
                "ISO 2023 listesi tablosunda 'NACE Kodu' = 29 olan tüm şirketler için "
                "toplam 'Ücretle Çalışanlar Ortalaması (Kişi)' değerini hesapla."
            ),
        },
        {
            "id": "iso_2024_tr",
            "lang": "TR",
            "category": "ISO_SQL_2024",
            "text": (
                "ISO 2024 listesi tablosunda 'ISIC Sektör Tanımı' değeri "
                "'Taşıt Araçları Sanayi' olan şirketler için ortalama 'FAVÖK (TL)' "
                "değerini hesapla."
            ),
        },
    ]

    # ---------------- Email drafting soruları ----------------

    en_email = [
        {
            "id": "email_sasa_2023_en",
            "lang": "EN",
            "category": "EMAIL_DRAFT",
            "text": (
                "Draft a professional email to the CEO summarizing the most important highlights "
                "from the Sasa Polyester 2023 annual report."
            ),
        },
        {
            "id": "email_iso_2022_en",
            "lang": "EN",
            "category": "EMAIL_DRAFT",
            "text": (
                "Draft a professional email to the head of quality assurance, sharing key observations "
                "from the ISO 2022 certification list."
            ),
        },
    ]

    tr_email = [
        {
            "id": "email_sasa_2023_tr",
            "lang": "TR",
            "category": "EMAIL_DRAFT",
            "text": (
                "Sasa Polyester'in 2023 faaliyet raporundaki en önemli başlıkları özetleyerek "
                "CEO'ya yönelik profesyonel bir e-posta taslağı hazırla."
            ),
        },
        {
            "id": "email_iso_2022_tr",
            "lang": "TR",
            "category": "EMAIL_DRAFT",
            "text": (
                "ISO 2022 sertifika listesinden çıkan temel gözlemleri paylaşan, "
                "kalite güvence müdürüne hitap eden profesyonel bir e-posta taslağı hazırla."
            ),
        },
    ]

    # ---------------- Genel sohbet ----------------

    en_chat = [
        {
            "id": "chat_rag_en",
            "lang": "EN",
            "category": "GENERAL_CHAT",
            "text": (
                "Explain in simple terms how this RAG system uses annual reports and ISO tables "
                "to answer executive questions."
            ),
        },
    ]

    tr_chat = [
        {
            "id": "chat_rag_tr",
            "lang": "TR",
            "category": "GENERAL_CHAT",
            "text": (
                "Bu RAG sisteminin faaliyet raporları ve ISO tablolarını kullanarak "
                "yönetim sorularını nasıl yanıtladığını basitçe açıkla."
            ),
        },
    ]

    # ---- Sıralama: önce tüm İngilizce (10), sonra tüm Türkçe (10) ----

    queries.extend(en_sasa)   # 3
    queries.extend(en_iso)    # +4 = 7
    queries.extend(en_email)  # +2 = 9
    queries.extend(en_chat)   # +1 = 10

    queries.extend(tr_sasa)   # 3
    queries.extend(tr_iso)    # +4 = 7
    queries.extend(tr_email)  # +2 = 9
    queries.extend(tr_chat)   # +1 = 10

    return queries


def run_test(output_name: str, runs: int = 5) -> Path:
    """Run the test queries for the given number of runs and write to text file."""
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = logs_dir / f"{output_name}_{timestamp}.txt"

    system = create_console_rag_system(db_config=None)

    # (İstersen) test sırasında konsol renkli loglarını kapat
    try:
        system.print_colored = lambda text, color_key="reset": None  # type: ignore
    except Exception:
        pass

    queries = build_test_queries()

    with output_path.open("w", encoding="utf-8") as f:
        header = (
            f"=== RAG System Test ===\n"
            f"Started at: {datetime.now().isoformat()}\n"
            f"Runs: {runs}\n"
            f"Total queries per run: {len(queries)} (10 EN + 10 TR)\n"
            f"{'='*80}\n\n"
        )
        f.write(header)

        for run_idx in range(1, runs + 1):
            f.write(f"##### RUN {run_idx} #####\n\n")
            for idx, q in enumerate(queries, start=1):
                t0 = time.time()
                try:
                    response = system.process_query(q["text"])
                except Exception as e:
                    response = f"ERROR: {str(e)}"
                elapsed = time.time() - t0

                f.write(
                    f"[Run {run_idx} | Query {idx}/{len(queries)}]\n"
                    f"ID       : {q['id']}\n"
                    f"Language : {q['lang']}\n"
                    f"Category : {q['category']}\n"
                    f"Elapsed  : {elapsed:.3f} s\n"
                    f"Question : {q['text']}\n"
                    f"Response :\n{response}\n"
                    f"{'-'*80}\n\n"
                )
                f.flush()

        f.write(f"=== Test finished at: {datetime.now().isoformat()} ===\n")

    try:
        system.close()
    except Exception:
        pass

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed test queries against the CEO RAG console system."
    )
    parser.add_argument(
        "output_name",
        help="Base name for the output log file (without extension).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of times to repeat the full query set (default: 5).",
    )

    args = parser.parse_args()

    try:
        output_path = run_test(output_name=args.output_name, runs=args.runs)
        print(f"Test completed. Results written to: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"Test failed: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

