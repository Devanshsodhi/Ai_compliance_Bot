import re

def parse_order_summary_text(text):
    data = {
        "type": "order_summary",   # added type
        "order_id": None,
        "shipping_details": {},
        "customer_details": {},
        "employee_details": {},
        "shipper_details": {},
        "order_details": {},
        "products": [],
        "total_price": None
    }

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    current_section = None
    current_product = {}

    for line in lines:
        if line.startswith("Order ID:"):
            data["order_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Shipping Details:"):
            current_section = "shipping"
        elif line.startswith("Customer Details:"):
            current_section = "customer"
        elif line.startswith("Employee Details:"):
            current_section = "employee"
        elif line.startswith("Shipper Details:"):
            current_section = "shipper"
        elif line.startswith("Order Details:"):
            current_section = "order"
        elif "Products:" in line:
            current_section = "products"
        elif line.startswith("Total Price:"):
            match = re.search(r"Total Price:\s*(\d+(\.\d+)?)", line)
            if match:
                data["total_price"] = float(match.group(1))

        elif current_section == "shipping":
            if "Ship Name:" in line:
                data["shipping_details"]["ship_name"] = line.split(":", 1)[1].strip()
            elif "Ship Address:" in line:
                data["shipping_details"]["ship_address"] = line.split(":", 1)[1].strip()
            elif "Ship City:" in line:
                data["shipping_details"]["ship_city"] = line.split(":", 1)[1].strip()
            elif "Ship Region:" in line:
                data["shipping_details"]["ship_region"] = line.split(":", 1)[1].strip()
            elif "Ship Postal Code:" in line:
                data["shipping_details"]["ship_postal_code"] = line.split(":", 1)[1].strip()
            elif "Ship Country:" in line:
                data["shipping_details"]["ship_country"] = line.split(":", 1)[1].strip()

        elif current_section == "customer":
            if "Customer ID:" in line:
                data["customer_details"]["customer_id"] = line.split(":", 1)[1].strip()
            elif "Customer Name:" in line:
                data["customer_details"]["customer_name"] = line.split(":", 1)[1].strip()

        elif current_section == "employee":
            if "Employee Name:" in line:
                data["employee_details"]["employee_name"] = line.split(":", 1)[1].strip()

        elif current_section == "shipper":
            if "Shipper ID:" in line:
                data["shipper_details"]["shipper_id"] = line.split(":", 1)[1].strip()
            elif "Shipper Name:" in line:
                data["shipper_details"]["shipper_name"] = line.split(":", 1)[1].strip()

        elif current_section == "order":
            if "Order Date:" in line:
                data["order_details"]["order_date"] = line.split(":", 1)[1].strip()
            elif "Shipped Date:" in line:
                data["order_details"]["shipped_date"] = line.split(":", 1)[1].strip()

        elif current_section == "products":
            if line.startswith("Product:"):
                if current_product:  
                    data["products"].append(current_product)
                    current_product = {}
                current_product["product_name"] = line.split(":", 1)[1].strip()
            elif "Quantity:" in line:
                current_product["quantity"] = int(line.split(":", 1)[1].strip())
            elif "Unit Price:" in line:
                current_product["unit_price"] = float(line.split(":", 1)[1].strip())
            elif "Total:" in line:
                current_product["total"] = float(line.split(":", 1)[1].strip())

    if current_product:
        data["products"].append(current_product)  # add last product

    return data


def parse_purchase_order_text(text, tables=None):
    lines = text.strip().splitlines()
    data = {
        "type": "purchase_order",  # added type
        "order_id": None,
        "order_date": None,
        "customer_name": None,
        "products": []
    }

    for i, line in enumerate(lines):
        line = line.strip()

        # Look for the line with Order ID, Date, and Customer Name
        if re.match(r'^\d{5}\s+\d{4}-\d{2}-\d{2}\s+.+', line):
            parts = line.split()
            if len(parts) >= 3:
                data["order_id"] = parts[0]
                data["order_date"] = parts[1]
                data["customer_name"] = " ".join(parts[2:])

        # Look for product lines after the header
        if "Product ID:" in line and "Product:" in line:
            # Start scanning the product lines
            for product_line in lines[i+1:]:
                product_line = product_line.strip()
                if not product_line or "Page" in product_line:
                    break  # stop at end or footer
                match = re.match(r'^(\d+)\s+(.*?)\s+(\d+)\s+([\d.]+)$', product_line)
                if match:
                    product_id, name, quantity, price = match.groups()
                    data["products"].append({
                        "product_id": product_id,
                        "product_name": name.strip(),
                        "quantity": int(quantity),
                        "unit_price": float(price)
                    })

            break  # no need to scan further

    return data


def parse_invoice_text(text, tables):
    lines = text.strip().splitlines()
    data = {
        "type": "invoice",  # added type
        "order_id": None,
        "customer_id": None,
        "order_date": None,
        "customer_details": {},
        "products": [],
        "total_price": None
    }

    # Extract key-value data from text
    for line in lines:
        line = line.strip()
        if line.startswith("Order ID:"):
            data["order_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Customer ID:"):
            data["customer_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Order Date:"):
            data["order_date"] = line.split(":", 1)[1].strip()
        elif line.startswith("Contact Name:"):
            data["customer_details"]["contact_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Address:"):
            data["customer_details"]["address"] = line.split(":", 1)[1].strip()
        elif line.startswith("City:"):
            data["customer_details"]["city"] = line.split(":", 1)[1].strip()
        elif line.startswith("Postal Code:"):
            data["customer_details"]["postal_code"] = line.split(":", 1)[1].strip()
        elif line.startswith("Country:"):
            data["customer_details"]["country"] = line.split(":", 1)[1].strip()
        elif line.startswith("Phone:"):
            data["customer_details"]["phone"] = line.split(":", 1)[1].strip()
        elif line.startswith("Fax:"):
            data["customer_details"]["fax"] = line.split(":", 1)[1].strip()
        elif line.startswith("TotalPrice"):
            match = re.search(r"\d+(\.\d+)?", line)
            if match:
                data["total_price"] = float(match.group())

    # Extract products from table (assumes 1 table holds product info)
    for tbl in tables:
        table = tbl["table"]
        if not table or len(table) < 2:
            continue  # skip empty or malformed tables

        headers = [h.lower() for h in table[0]]
        for row in table[1:]:
            if len(row) >= 4:
                try:
                    product = {
                        "product_id": row[0].strip(),
                        "product_name": row[1].strip(),
                        "quantity": int(row[2]),
                        "unit_price": float(row[3])
                    }
                    data["products"].append(product)
                except (ValueError, IndexError):
                    continue  # skip invalid rows

    return data
