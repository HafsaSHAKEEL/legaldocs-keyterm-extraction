
# LegalDocs Term Extraction and Evaluation

## Overview

This project is designed to extract key terms from legal documents, map these terms to specific sections within the documents, and evaluate the extraction performance against a set of ground truth terms. The system uses YAKE for keyword extraction and includes functionalities for preprocessing text, ensuring the presence of critical terms, filtering relevant terms, and evaluating precision, recall, and mapping efficiency.

## Directory Structure

```
legaldpcs/
├── .venv/
│   ├── bin/
│   ├── lib/
│   ├── .gitignore
│   └── pyvenv.cfg
├── extracted_texts/
├── ground_truth_terms/
├── legaldocs/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── main.py
│   ├── pdf2text.py
│   ├── preprocess.py
├── pdf/
│   ├── employment-contract-revised.pdf
├── term_extraction_results/
│   ├── .gitignore
│   ├── pdf2text.py
│   ├── preprocess.py
├── requirements.txt
└── update.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legaldocs.git
   cd legaldocs
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## Usage

### Extracting Text from PDFs

1. Place your PDF files in the `pdf/` directory.
2. Run the `pdf2text.py` script to extract text from the PDFs:
   ```bash
   python legaldocs/pdf2text.py
   ```
   This will save the extracted text files in the `extracted_texts/` directory.

### Preprocessing and Extracting Key Terms

1. Place your ground truth term files in the `ground_truth_terms/` directory.
2. Run the `main.py` script to preprocess the text files, extract key terms, and map these terms to sections:
   ```bash
   python -m legaldocs.main
   ```
   This will save the term extraction results in the `term_extraction_results/` directory.

### Evaluating Term Extraction Performance

1. Run the `evaluate.py` script to evaluate the performance of the term extraction:
   ```bash
   python -m legaldocs.evaluate
   ```
   This script will output precision, recall, and mapping efficiency metrics for each document.

### Updating Term Extraction Results

1. Run the `update.py` script to monitor changes and update term extraction results:
   ```bash
   python legaldocs/update.py
   ```
   This script will continuously monitor the term extraction results directory for changes and update the results accordingly.

## Scripts

### `main.py`

Handles the entire term extraction process including preprocessing, extracting key terms, ensuring critical terms, and saving the results. Execute it using:
```bash
python -m legaldocs.main
```

### `pdf2text.py`

Extracts text from PDF files using PyMuPDF and performs various preprocessing steps to clean and format the text.

### `preprocess.py`

Contains the `TermExtractionHandler` class which includes methods for loading text files, extracting key terms, ensuring critical and ground truth terms, mapping terms to sections, and saving term extraction results.

### `evaluate.py`

Evaluates the performance of the term extraction process by calculating precision, recall, and mapping efficiency metrics. Execute it using:
```bash
python -m legaldocs.evaluate
```

### `update.py`

Monitors the term extraction results directory for changes and updates the results accordingly.

## Contact

For any questions or issues, please contact [yourname@yourdomain.com](mailto:yourname@yourdomain.com).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This version includes the corrected usage instructions and ensures that relative imports are properly handled within the project structure.
