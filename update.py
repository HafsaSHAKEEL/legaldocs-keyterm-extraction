import os
import re
import hashlib
import logging
import time
from pathlib import Path
from preprocess import TermExtractionHandler  # Reuse the TermExtractionHandler class

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TermExtractionUpdater(TermExtractionHandler):
    """
    Class to handle updating term extraction results by monitoring text files and extracting key terms.
    """

    def __init__(self, text_directory, results_directory):
        super().__init__(results_directory)
        self.text_directory = text_directory
        self.term_extraction_results = self.load_term_extraction_results()
        self.file_hashes = self.compute_initial_hashes()

    def compute_file_hash(self, file_path):
        """
        Compute the MD5 hash of a file.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: The MD5 hash of the file.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        logging.info(f"Hash for {file_path}: {hash_md5.hexdigest()}")
        return hash_md5.hexdigest()

    def compute_initial_hashes(self):
        """
        Compute initial hashes for all files in the results directory.

        Returns:
            dict: A dictionary mapping file paths to their hashes.
        """
        file_hashes = {}
        for filename in os.listdir(self.results_directory):
            if filename.endswith("_terms.txt"):
                file_path = os.path.join(self.results_directory, filename)
                file_hashes[file_path] = self.compute_file_hash(file_path)
        logging.info(f"Initial Hashes: {file_hashes}")
        return file_hashes

    def load_term_extraction_results(self):
        """
        Load term extraction results from a directory.

        Returns:
            dict: A dictionary where keys are filenames and values are dictionaries with terms and term_section_map.
        """
        results = {}
        for filename in os.listdir(self.results_directory):
            if filename.endswith("_terms.txt"):
                with open(
                    os.path.join(self.results_directory, filename),
                    "r",
                    encoding="utf-8",
                ) as file:
                    content = file.read()
                    terms, term_section_map = self.parse_term_extraction_results(content)
                    results[filename] = {
                        "terms": terms,
                        "term_section_map": term_section_map,
                    }
        logging.info(f"Loaded Results: {results}")
        return results

    def parse_term_extraction_results(self, content):
        """
        Parse the term extraction results from the file content.

        Args:
            content (str): Content of the term extraction results file.

        Returns:
            tuple: A tuple containing a list of terms and a dictionary mapping terms to their sections.
        """
        lines = content.splitlines()
        terms = set()
        term_section_map = {}
        current_term = None

        for line in lines:
            if line.startswith("- "):
                term = line[2:].strip()
                terms.add(term)
                term_section_map[term] = []
                current_term = term
            elif line.startswith("Term:"):
                current_term = line.split(":")[1].strip()
            elif line.startswith("Section:"):
                if current_term:
                    term_section_map[current_term].append(line[9:].strip())

        logging.info(f"Parsed Terms: {terms}")
        logging.info(f"Term-Section Map: {term_section_map}")
        return list(terms), term_section_map

    def display_available_documents(self):
        """
        Display the available documents.
        """
        logging.info("\nAvailable documents:")
        for idx, filename in enumerate(self.term_extraction_results.keys(), start=1):
            logging.info(f"{idx}. {filename}")

    def display_terms(self, terms, term_section_map):
        """
        Display the extracted terms and their mapped sections.

        Args:
            terms (list): List of terms.
            term_section_map (dict): Dictionary mapping terms to sections.
        """
        logging.info("Extracted Key Terms:")
        for term in terms:
            if term in term_section_map:
                logging.info(f"- {term}")
        logging.info("\nMapped Sections:")
        for term, sections in term_section_map.items():
            logging.info(f"\nTerm: {term}")
            for section in sections:
                logging.info(f"Section: {section}\n---")

    def display_updated_document(self, filename):
        """
        Display the updated document's terms and sections.

        Args:
            filename (str): Name of the file to display.
        """
        result = self.term_extraction_results[filename]
        self.display_terms(result["terms"], result["term_section_map"])

    def monitor_changes(self):
        """
        Monitor the term extraction results directory for changes.
        """
        while True:
            time.sleep(1)
            for file_path, old_hash in self.file_hashes.items():
                new_hash = self.compute_file_hash(file_path)
                if new_hash != old_hash:
                    logging.info(f"Detected modification in {file_path}. Updating results...")
                    self.file_hashes[file_path] = new_hash
                    self.term_extraction_results = self.load_term_extraction_results()
                    self.display_updated_document(os.path.basename(file_path))
                else:
                    logging.info(f"No actual change detected in {file_path}.")
            logging.info("Stopping observer...")

if __name__ == "__main__":
    text_directory = "extracted_texts"
    results_directory = "term_extraction_results"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    updater = TermExtractionUpdater(text_directory, results_directory)
    updater.display_available_documents()
    updater.monitor_changes()
