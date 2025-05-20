import pdfplumber

def extract_text_and_tables_from_pdf(pdf_path):
    full_text = ""
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_number} ---\n{text}"
            
            tables = page.extract_tables()
            for table in tables:
                all_tables.append({
                    "page": page_number,
                    "table": table
                })

    return {
        "text": full_text.strip(),
        "tables": all_tables
    }
