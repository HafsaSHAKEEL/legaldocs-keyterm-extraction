import os
import logging
from fuzzywuzzy import fuzz
from sklearn.metrics import precision_score, recall_score


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_terms_and_map(directory):
    """
    Load terms and their corresponding sections from files in the specified directory.

    Args:
        directory (str): Path to the directory containing the term extraction results.

    Returns:
        dict: A dictionary where keys are filenames and values are tuples containing
              a list of terms and a dictionary mapping terms to their sections.
    """
    terms_map = {}
    for filename in os.listdir(directory):
        if filename.endswith("_terms.txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                content = file.read()
                terms, term_section_map = parse_term_extraction_results(content)
                terms_map[filename] = (terms, term_section_map)
                logging.info(f"Loaded terms from {filename}")
    return terms_map


def parse_term_extraction_results(content):
    """
    Parse the term extraction results from file content.

    Args:
        content (str): Content of the term extraction results file.

    Returns:
        tuple: A tuple containing a list of terms and a dictionary mapping terms to their sections.
    """
    lines = content.splitlines()
    terms = []
    term_section_map = {}
    current_term = None

    for line in lines:
        if line.startswith("- "):
            term = line[2:].strip().lower()  # Convert to lowercase for consistency
            terms.append(term)
            term_section_map[term] = []
            current_term = term
        elif line.startswith("Term:"):
            current_term = (
                line.split(":")[1].strip().lower()
            )  # Convert to lowercase for consistency
        elif line.startswith("Section:"):
            if current_term:
                term_section_map[current_term].append(
                    line[9:].strip().lower()
                )  # Convert to lowercase for consistency

    return terms, term_section_map


def evaluate_precision_recall(ground_truth_terms, extracted_terms):
    """
    Evaluate the precision and recall of the extracted terms compared to the ground truth terms.

    Args:
        ground_truth_terms (list): List of ground truth terms.
        extracted_terms (list): List of extracted terms.

    Returns:
        tuple: A tuple containing precision and recall scores.
    """
    all_terms = list(set(ground_truth_terms + extracted_terms))
    y_true = [term in ground_truth_terms for term in all_terms]
    y_pred = [term in extracted_terms for term in all_terms]

    logging.info(f"Ground Truth Terms: {ground_truth_terms}")
    logging.info(f"Extracted Terms: {extracted_terms}")

    true_positives = sum(y_true[i] and y_pred[i] for i in range(len(y_true)))
    false_positives = sum(y_pred[i] and not y_true[i] for i in range(len(y_true)))
    false_negatives = sum(y_true[i] and not y_pred[i] for i in range(len(y_true)))

    logging.info(f"True Positives: {true_positives}")
    logging.info(f"False Positives: {false_positives}")
    logging.info(f"False Negatives: {false_negatives}")

    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)

    return precision, recall


def evaluate_mapping_efficiency(ground_truth_map, extracted_map):
    """
    Evaluate the mapping efficiency of the extracted terms to their sections compared to the ground truth.

    Args:
        ground_truth_map (dict): Dictionary mapping ground truth terms to their sections.
        extracted_map (dict): Dictionary mapping extracted terms to their sections.

    Returns:
        float: The mapping efficiency score.
    """
    correct_mappings = 0
    total_mappings = 0

    for term, sections in extracted_map.items():
        if term in ground_truth_map:
            for extracted_section in sections:
                best_similarity = 0
                for ground_truth_section in ground_truth_map[term]:
                    ratio_similarity = fuzz.ratio(
                        extracted_section, ground_truth_section
                    )
                    partial_similarity = fuzz.partial_ratio(
                        extracted_section, ground_truth_section
                    )
                    token_set_similarity = fuzz.token_set_ratio(
                        extracted_section, ground_truth_section
                    )
                    similarity = max(
                        ratio_similarity, partial_similarity, token_set_similarity
                    )
                    best_similarity = max(best_similarity, similarity)

                if best_similarity > 75:  # Lowered threshold for better matching
                    correct_mappings += 1
                total_mappings += 1

            logging.info(f"Term: {term}")
            logging.info(f"Extracted Sections: {sections}")
            logging.info(f"Ground Truth Sections: {ground_truth_map[term]}")

    mapping_efficiency = correct_mappings / total_mappings if total_mappings > 0 else 0

    logging.info(f"Correct Mappings: {correct_mappings}")
    logging.info(f"Total Mappings: {total_mappings}")
    logging.info(f"Mapping Efficiency: {mapping_efficiency:.4f}")

    return mapping_efficiency


def main():
    """
    Main function to load ground truth terms and extracted results, and evaluate the performance.
    """
    ground_truth_directory = "ground_truth_terms"
    results_directory = "term_extraction_results"

    logging.info("Loading ground truth terms...")
    ground_truth = load_terms_and_map(ground_truth_directory)

    logging.info("Loading extracted results...")
    extracted_results = load_terms_and_map(results_directory)

    logging.info("Starting evaluation...")
    for filename, (ground_truth_terms, ground_truth_map) in ground_truth.items():
        if filename in extracted_results:
            extracted_terms, extracted_map = extracted_results[filename]

            precision, recall = evaluate_precision_recall(
                ground_truth_terms, extracted_terms
            )
            mapping_efficiency = evaluate_mapping_efficiency(
                ground_truth_map, extracted_map
            )

            logging.info(f"Evaluation for {filename}:")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"Mapping Efficiency: {mapping_efficiency:.4f}")
            logging.info("\n---\n")


if __name__ == "__main__":
    main()
