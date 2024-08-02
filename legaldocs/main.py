import os
import logging
from dotenv import load_dotenv
from legaldocs.preprocess import TermExtractionHandler
from legaldocs.evaluate import main as evaluate_main
from legaldocs.pdf2text import extract_text_from_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess():
    pdf_directory = "pdf"
    text_directory = "extracted_texts"
    results_directory = "term_extraction_results"
    ground_truth_directory = "ground_truth_terms"

    # Ensure directories exist
    os.makedirs(pdf_directory, exist_ok=True)
    os.makedirs(text_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(ground_truth_directory, exist_ok=True)

    # Extract text from PDFs
    extract_text_from_pdf(pdf_directory, text_directory)

    handler = TermExtractionHandler(results_directory)
    ground_truth_terms = handler.load_ground_truth_terms(ground_truth_directory)
    handler.process_files(text_directory, results_directory, ground_truth_terms)

def evaluate():
    evaluate_main()

def main():
    logging.info("Starting preprocessing...")
    preprocess()

    logging.info("Starting evaluation...")
    evaluate()

if __name__ == "__main__":
    main()
