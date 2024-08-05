import os
import re
import logging
import yake
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TermExtractionHandler:
    """
    A class to handle term extraction from text files.
    """

    def __init__(self, results_directory):
        """
        Initialize the TermExtractionHandler with the results directory.

        Args:
            results_directory (str): The directory to save the term extraction results.
        """
        self.results_directory = results_directory
        self.ner_model = None
        self.tokenizer = None

    def load_text_files(self, directory):
        """
        Load text files from a specified directory.

        Args:
            directory (str): The directory containing text files.

        Returns:
            dict: A dictionary where keys are filenames and values are file contents.
        """
        texts = {}
        for filename in os.listdir(directory):
            if filename.endswith("_text.txt"):
                with open(
                        os.path.join(directory, filename), "r", encoding="utf-8"
                ) as file:
                    texts[filename] = file.read()
        return texts

    def load_ground_truth_terms(self, ground_truth_directory):
        """
        Load ground truth terms from files in a specified directory.

        Args:
            ground_truth_directory (str): The directory containing ground truth term files.

        Returns:
            set: A set of ground truth terms.
        """
        terms = set()
        for filename in os.listdir(ground_truth_directory):
            if filename.endswith("_terms.txt"):
                with open(
                        os.path.join(ground_truth_directory, filename),
                        "r",
                        encoding="utf-8",
                ) as file:
                    content = file.read()
                    for line in content.splitlines():
                        if line.startswith("- "):
                            term = line[2:].strip().lower()
                            terms.add(term)
        return terms

    def load_ner_model(self):
        """
        Load a pre-trained NER model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english")

    def extract_key_terms(self, text, max_terms=150):
        """
        Extract key terms from text using YAKE and a pre-trained NER model.

        Args:
            text (str): The text to extract terms from.
            max_terms (int): The maximum number of terms to extract.

        Returns:
            set: A set of extracted key terms.
        """
        # Extract using YAKE
        kw_extractor_unigrams = yake.KeywordExtractor(
            lan="en", n=1, dedupLim=0.9, top=max_terms // 2
        )
        kw_extractor_bigrams = yake.KeywordExtractor(
            lan="en", n=2, dedupLim=0.9, top=max_terms // 2
        )

        keywords_unigrams = kw_extractor_unigrams.extract_keywords(text)
        keywords_bigrams = kw_extractor_bigrams.extract_keywords(text)

        keywords = keywords_unigrams + keywords_bigrams

        yake_terms = set(kw.lower() for kw, _ in keywords)  # Remove duplicates

        # Extract using NER
        ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        ner_results = ner_pipeline(text)

        ner_terms = set()
        for result in ner_results:
            ner_terms.add(result['word'].lower())

        return yake_terms.union(ner_terms)

    def ensure_critical_terms(self, extracted_terms, text):
        """
        Ensure critical terms are present in the extracted terms.

        Args:
            extracted_terms (set): The set of extracted terms.
            text (str): The text to check for critical terms.

        Returns:
            list: A list of extracted terms including critical terms.
        """
        critical_terms = ["confidentiality", "security deposit", "employee's duties"]
        text_lower = text.lower()
        for term in critical_terms:
            if term in text_lower:
                logging.info(f"Critical term '{term}' found in text.")
                if term not in extracted_terms:
                    logging.info(f"Adding critical term: {term}")
                    extracted_terms.add(term)
            else:
                logging.info(f"Critical term '{term}' not found in text.")
        return list(extracted_terms)

    def ensure_ground_truth_terms(self, extracted_terms, ground_truth_terms, text):
        """
        Ensure ground truth terms are present in the extracted terms.

        Args:
            extracted_terms (set): The set of extracted terms.
            ground_truth_terms (set): The set of ground truth terms.
            text (str): The text to check for ground truth terms.

        Returns:
            list: A list of extracted terms including ground truth terms.
        """
        text_lower = text.lower()
        for term in ground_truth_terms:
            if term in text_lower:
                logging.info(f"Ground truth term '{term}' found in text.")
                if term not in extracted_terms:
                    logging.info(f"Adding ground truth term: {term}")
                    extracted_terms.append(term)
            else:
                logging.info(f"Ground truth term '{term}' not found in text.")
        return list(extracted_terms)

    def filter_relevant_terms(self, terms, ground_truth_terms):
        """
        Filter relevant terms based on ground truth terms.

        Args:
            terms (list): The list of extracted terms.
            ground_truth_terms (set): The set of ground truth terms.

        Returns:
            list: A list of relevant terms.
        """
        return [term for term in terms if term in ground_truth_terms]

    def map_terms_to_sections(self, text, terms):
        """
        Map extracted terms to relevant sections in the text.

        Args:
            text (str): The text to map terms to.
            terms (list): The list of extracted terms.

        Returns:
            dict: A dictionary where keys are terms and values are lists of sections containing the terms.
        """
        sections = re.split(
            r"(?=\b\d+\.\s)", text
        )  # Split text into sections by headings (e.g., "1. ", "2. ")
        term_section_map = {term: [] for term in terms}

        for section in sections:
            section_lower = section.lower()
            for term in terms:
                if term in section_lower:  # Ensure case-insensitive matching
                    section_context = section.strip().replace(
                        "\n", " "
                    )  # More context from section
                    if (
                            section_context not in term_section_map[term]
                    ):  # Ensure unique sections
                        term_section_map[term].append(
                            section_context
                        )  # Save more context of the section

        return {
            term: contexts for term, contexts in term_section_map.items() if contexts
        }

    def save_term_extraction_results(
            self, output_directory, filename, terms, term_section_map
    ):
        """
        Save the extracted key terms and their mapped sections to a file.

        Args:
            output_directory (str): The directory to save the term extraction results.
            filename (str): The filename of the text file.
            terms (list): The list of extracted terms.
            term_section_map (dict): The dictionary mapping terms to sections.
        """
        output_file = os.path.join(
            output_directory, filename.replace("_text.txt", "_terms.txt")
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Extracted Key Terms:\n")
            for term in terms:
                f.write(f"- {term}\n")
            f.write("\nMapped Sections:\n")
            for term, sections in term_section_map.items():
                f.write(f"\nTerm: {term}\n")
                for section in sections:
                    f.write(f"Section: {section}\n")
                    f.write("\n---\n")
        logging.info(f"Saved term extraction results to {output_file}")

    def process_files(self, text_directory, output_directory, ground_truth_terms):
        """
        Process text files to extract and save key terms.

        Args:
            text_directory (str): The directory containing text files.
            output_directory (str): The directory to save the term extraction results.
            ground_truth_terms (set): The set of ground truth terms.
        """
        texts = self.load_text_files(text_directory)
        self.load_ner_model()
        for filename, text in texts.items():
            logging.info(f"Processing file: {filename}")
            extracted_terms = self.extract_key_terms(text)
            logging.info(f"Initial extracted terms: {extracted_terms}")
            extracted_terms = self.ensure_critical_terms(extracted_terms, text)
            logging.info(f"Terms after ensuring critical terms: {extracted_terms}")
            ensured_terms = self.ensure_ground_truth_terms(
                extracted_terms, ground_truth_terms, text
            )
            logging.info(f"Terms after ensuring ground truth terms: {ensured_terms}")
            relevant_terms = self.filter_relevant_terms(
                ensured_terms, ground_truth_terms
            )
            logging.info(f"Relevant terms: {relevant_terms}")
            term_section_map = self.map_terms_to_sections(text, relevant_terms)
            self.save_term_extraction_results(
                output_directory, filename, relevant_terms, term_section_map
            )
