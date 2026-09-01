import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import requests
from mcps.sec_edgar import get_cik, get_filings, download_filing_text

cik = get_cik("AAPL")
print("CIK:", cik)

filings = get_filings(cik, count=3)
for f in filings:
    print(f["date"], f["accession_number"])

text = None
for f in filings:
    try:
        text = download_filing_text(f["accession_number"], cik, f["primary_document"])
        print(f"\nUsing filing {f['accession_number']} ({f['date']})")
        break
    except requests.exceptions.HTTPError as e:
        print(f"Skipping {f['accession_number']}: {e}")

if text is None:
    raise RuntimeError("No filing could be downloaded.")

print(f"\nGot {len(text):,} chars")
print(text[:500])