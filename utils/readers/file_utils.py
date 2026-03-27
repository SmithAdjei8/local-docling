import json
import logging

def read_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp1252') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

def read_json(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
        raise

def write_json(file_path: str, data: dict):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Error writing JSON to file {file_path}: {e}")
        raise

def write_file(file_path: str, content: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logging.error(f"Error writing to file {file_path}: {e}")
        raise