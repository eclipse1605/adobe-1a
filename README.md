# PDF Title and Heading Extractor

This project extracts titles and headings from PDF documents using a machine learning-based approach. It processes a directory of PDF files and generates structured JSON output for each, outlining the document's hierarchy.

## Approach

The core of this solution is a text classification model that identifies the structural role of each line in a PDF, such as "title," "heading," or "paragraph." The process can be broken down into the following steps:

1.  **PDF Parsing**: The `PyMuPDF` library is used to open and read each PDF document. It extracts text blocks along with their properties, including font size, font name, and position on the page.

2.  **Feature Engineering**: For each extracted text block, a set of features is generated to help the model distinguish between different types of content. These features include:
    *   Text content and length
    *   Font size and style
    *   Vertical and horizontal positioning
    *   Spacing relative to adjacent text blocks

3.  **Heading Prediction**: A pre-trained LightGBM classification model (`title_detection_model.joblib`) predicts the category for each text block (e.g., `title`, `H1`, `H2`, `H3`, or `other`).

4.  **JSON Output Generation**: The detected titles and headings are organized into a hierarchical JSON structure that reflects the document's outline. The output is saved as a `.json` file corresponding to each input PDF.

## Models and Libraries

### Models

*   **`title_detection_model.joblib`**: A pre-trained LightGBM (Light Gradient Boosting Machine) model for classifying text blocks.
*   **`heading_detection_model.joblib`**: A pre-trained LightGBM model for classifying heading levels.

These models are located in the `/models` directory.

### Key Libraries

*   **`scikit-learn`**: Used to load the pre-trained machine learning model and associated artifacts.
*   **`pandas`**: For efficient data manipulation and structuring of the extracted text features.
*   **`numpy`**: For numerical operations.
*   **`lightgbm`**: The library that provides the gradient boosting framework used for the classification models.
*   **`PyMuPDF`**: For high-performance PDF parsing and text extraction.
*   **`fuzzywuzzy`**: Used for string matching and text processing.
*   **`python-Levenshtein`**: Provides a fast implementation of Levenshtein distance calculations, which can be useful in text comparisons.

A full list of dependencies is available in `requirements.txt`.

## How to Build and Run

The solution is delivered as a Dockerized application. The following instructions describe how to build the Docker image and run the container to process your PDF files.

### Prerequisites

*   Docker must be installed on your system.

### 1. Build the Docker Image

Navigate to the root directory of the project and run the following command to build the Docker image.

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### 2. Prepare Input Files

Place all the PDF files you want to process into the `input` directory in the project root.

### 3. Run the Container

Execute the following command to run the container. This command mounts the `input` and `output` directories from your local machine to the container, allowing it to read the PDFs and write the JSON results.

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

The container will automatically process all PDF files in the `/app/input` directory and generate a corresponding `.json` file for each in the `/app/output` directory. 