import os
import pandas as pd
import pdfplumber
import docx
from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------- PDF Loader --------
def load_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -------- DOCX Loader --------
def load_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    return "\n".join([para.text for para in document.paragraphs])


# -------- CSV Loader --------
def load_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_markdown(index=False)


# -------- TXT Loader --------
def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------- Image Loader (OCR) --------
def load_image(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text


# -------- Main Dispatcher --------
def load_document(file_path: str) -> str:
    """
    Detects file type and calls the appropriate loader.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension in [".docx", ".doc"]:
        return load_docx(file_path)

    elif extension == ".csv":
        return load_csv(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    elif extension in [".png", ".jpg", ".jpeg"]:
        return load_image(file_path)

    else:
        raise ValueError(f"Unsupported file format: {extension}")


# -------- Simple Test --------
if __name__ == "__main__":
    sample_path = "data/sample_docs/sample.txt"
    text = load_document(sample_path)
    print("Document Loaded Successfully!")
    print(text[:500])
