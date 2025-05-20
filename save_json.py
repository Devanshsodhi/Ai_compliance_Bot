import os
import json

def save_document_data(doc_type, new_data, output_folder):
    filename = f"{doc_type.lower().replace(' ', '_')}.json"
    file_path = os.path.join(output_folder, filename)

    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except Exception:
                existing_data = []
    else:
        existing_data = []

    
    unique_key = "order_id" if "order_id" in new_data else None

    if unique_key:
        
        existing_index = None
        for i, entry in enumerate(existing_data):
            if entry.get(unique_key) == new_data.get(unique_key):
                existing_index = i
                break
        
        if existing_index is not None:
            
            existing_data[existing_index] = new_data
            print(f"♻️ Updated existing {doc_type} with {unique_key}={new_data.get(unique_key)}")
        else:
           
            existing_data.append(new_data)
            print(f"➕ Added new {doc_type} with {unique_key}={new_data.get(unique_key)}")
    else:
        
        existing_data.append(new_data)
        print(f"➕ Added new {doc_type} (no unique key)")

    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
