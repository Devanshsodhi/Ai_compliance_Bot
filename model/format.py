from langchain.schema import Document
import os
import json

def load_json_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            documents.append(
                                Document(
                                    page_content=json.dumps(item, indent=2),
                                    metadata={"source": filename}
                                )
                            )
                    elif isinstance(data, dict):
                        documents.append(
                            Document(
                                page_content=json.dumps(data, indent=2),
                                metadata={"source": filename}
                            )
                        )
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
    return documents
