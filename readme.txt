# SMARTBatch Project

## Overview

The company's factories have been experiencing significant inefficiencies due to frequent adjustments required during the production phase of certain formulas. These adjustments have led to increased production time, higher resource consumption, and inconsistencies in product quality. To address these challenges, the SMARTBatch project was launched with the goal of developing a machine learning system that recommends new formula references, thereby minimizing the need for batch adjustments. This initiative aims to enhance production stability and consistency across all L’Oréal factories.

## Prerequisites

### Software Requirements

- Python 3.11.1
- Jupyter Notebook

### Installation Instructions

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required libraries from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### Data Files

The project requires the following data files, provided by the business partners:

- **Batches History Data**: `IA BPC Teinte v2.xlsm`
- **Requested Formulas Data**: `IA BPC Teinte Besoins.xlsm`
- **Reference Formulas Data**: `composition des SOP avec type d'intro MP.xlsm`

These files should be placed in a folder named with the data date, e.g., `data 19-08-2024`.

### Configuration

Before running any of the notebooks, configure the following variables:

- **Data Date**: Enter the date of the provided data in the format `DD-MM-YYYY`.
- **Starting Date for Grid Search**: Specify the starting date for historical data as `LIMIT_INITIALIZATION_DATE` in the format `YYYY-MM-DD`. # modifier
- **Category Name for Export Files**: Provide the category name for export file naming as `category_name_for_export_files` (relevant for the 'main_selected_category' notebook).
- **Raw Material Categories**: Specify the list of raw material categories as `SELECTED_MP_CATEGORIES` (relevant for the 'main_selected_category' notebook).

### Running the Notebook

1. **Launch Jupyter Notebook**: Open your terminal or command prompt and run:
    ```bash
    jupyter notebook
    ```

2. **Open the Notebook**: Navigate to the directory containing the notebook and open it.

3. **Execute the Cells**: Run each cell sequentially, following any prompts for user input.

## Outputs

The results are saved in an Excel file with the following sheets:

- **Formulas Predictions**: Contains detailed predictions of formula references compared to the provided references.
- **Formulas Indicators**: Includes information about formulas, production requests, stability status, grid search results, and summary recommendations.
- **Parameters and Metrics**: Details the parameters and metrics used in the filtering and grid search processes.

## Contact

For further assistance, please contact Aurelien Zakowic, Senior Data Scientist at Tech Accelerator Paris.
