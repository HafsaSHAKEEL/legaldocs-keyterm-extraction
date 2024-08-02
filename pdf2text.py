import os
import logging
import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_text_from_pdf(pdf_directory, text_directory):
    """
    Extract text from all PDF files in the specified directory and save them as text files.

    Args:
        pdf_directory (str): The directory containing PDF files.
        text_directory (str): The directory to save the extracted text files.
    """
    if not os.path.exists(pdf_directory):
        logging.error(f"Error processing {pdf_directory}: no such file or directory.")
        return

    if not os.path.exists(text_directory):
        os.makedirs(text_directory)

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text_path = os.path.join(text_directory, filename.replace(".pdf", "_text.txt"))

            try:
                logging.info(f"Extracting text from {pdf_path}")
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()

                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)
                logging.info(f"Saved extracted text to {text_path}")
            except Exception as e:
                logging.error(f"Error extracting text from {pdf_path}: {e}")
