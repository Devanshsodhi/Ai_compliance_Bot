import os
import sys
import json
import warnings
import contextlib
from extraction.extract import extract_text_and_tables_from_pdf
from parser.unified_parser import parse_invoice_text
from parser.unified_parser import parse_purchase_order_text
from parser.unified_parser import parse_order_summary_text
from save_json import save_document_data

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def detect_document_type(text):
    lowered = text.lower()
    if "invoice" in lowered:
        return "invoice"
    elif "purchase order" in lowered or "purchase orders" in lowered:
        return "purchase_order"
    elif "shipping details" in lowered and "order details" in lowered:
        return "order_summary"
    return "unknown"

def load_existing_order_ids(output_folder):
    order_ids = set()
    for fname in ["invoice.json", "purchase_order.json", "order_summary.json"]:
        path = os.path.join(output_folder, fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry in data:
                        if "order_id" in entry:
                            order_ids.add(str(entry["order_id"]))
            except Exception:
                continue
    return order_ids

def main():
    input_folder = "input_folder"
    output_folder = "output_folder"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(input_folder, filename)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with suppress_stdout_stderr():
                    result = extract_text_and_tables_from_pdf(path)

            doc_type = detect_document_type(result["text"])
            parsed = None

            if doc_type == "invoice":
                parsed = parse_invoice_text(result["text"], result["tables"])
            elif doc_type == "purchase_order":
                parsed = parse_purchase_order_text(result["text"], result["tables"])
            elif doc_type == "order_summary":
                parsed = parse_order_summary_text(result["text"])
            else:
                continue

            if parsed:
                parsed["type"] = doc_type
                save_document_data(doc_type.replace('_', ' '), parsed, output_folder)



if __name__ == "__main__":
    main()
