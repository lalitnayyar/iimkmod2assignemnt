# Titanic Dataset Analysis - Assignment 2.1

## Course Information
- **Course:** IIMK's Professional Certificate in Data Science and Artificial Intelligence for Managers
- **Assignment:** Week 2: Required Assignment 2.1
- **Submitted By:** Lalit Nayyar
- **Email:** lalitnayyar@gmail.com

## Project Overview

A comprehensive data analysis and preparation solution for the Titanic dataset, featuring professional-grade visualizations, detailed reporting, and thorough documentation of all decisions and insights.

## Professional Assignment Section (NEW)

The HTML report now features a dedicated **Assignment Part: Data Analysis & Preparation** section at the top, addressing all required questions with:
- **Four detailed subsections:**
  1. Understanding Raw Data
  2. Data Cleaning Techniques
  3. Data Transformation
  4. Reflection and Insights
- **Professional formatting** with icons, tables, and clear explanations for each question.
- **Key highlights:**
  - Table of columns, types, missing values, and descriptions
  - Explicit answers on dataset structure, missing values, duplicates, and common issues
  - Cleaning and transformation strategies with justifications
  - Reflection on challenges and importance of each step

**To view this section:**
- Open `output/analysis_report.html` in your browser.
- The Assignment section is at the top, visually distinct and easy to navigate.

## Table of Contents
1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [User Guide](#user-guide)
4. [Analysis Components](#analysis-components)
5. [Output Files](#output-files)
6. [Implementation Details](#implementation-details)
7. [Submission Details](#submission-details)

## Features

### 1. Data Understanding
- Complete dataset structure analysis
- Missing value detection and visualization
- Duplicate record identification
- Descriptive statistics generation
- Interactive visualizations

### 2. Data Cleaning
- Sophisticated missing value handling
- Feature selection with justification
- Data quality improvements
- Correlation analysis

### 3. Data Transformation
- Advanced categorical encoding
- Professional numerical scaling
- Distribution analysis
- Feature engineering

### 4. Professional Reporting
- Interactive HTML reports
- Publication-quality visualizations
- Detailed JSON insights
- Comprehensive logging

## Setup Instructions

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Kaggle API credentials:
   - Create a Kaggle account at https://www.kaggle.com
   - Go to your account settings (https://www.kaggle.com/account)
   - Scroll to 'API' section and click 'Create New API Token'
   - This will download 'kaggle.json'
   - Place this file in:
     - Windows: `%USERPROFILE%\.kaggle\`

3. Download the dataset:
   ```bash
   python download_dataset.py
   ```

## User Guide

### Running the Analysis

1. Execute the main analysis script:
   ```bash
   python Lalit_Nayyar_titanic_analysis.py
   ```

2. View the Results:
   - Open `output/analysis_report.html` in a web browser
   - Review generated visualizations in `output/plots/`
   - Check detailed insights in `output/analysis_insights.json`

### Understanding the Output

1. **HTML Report Sections:**
   - Data Understanding
   - Missing Values Analysis
   - Feature Correlations
   - Transformed Distributions
   - Code Implementation Details
   - Final Conclusions

2. **Visualizations:**
   - Data distribution plots
   - Missing values heatmap
   - Correlation matrix
   - Feature distributions

3. **Analysis Results:**
   - Processed dataset
   - Technical insights
   - Statistical summaries
   - Implementation decisions

## Analysis Components

### 1. Data Understanding
- Dataset structure analysis
- Column type identification
- Missing value patterns
- Data quality assessment
- Initial visualizations

### 2. Data Cleaning
- Missing value imputation strategies:
  - Age: Stratified median imputation
  - Cabin: Binary feature conversion
  - Embarked: Mode imputation
- Feature selection with impact analysis
- Quality validation

### 3. Data Transformation
- Categorical encoding:
  - Label encoding for binary features
  - Ordinal encoding for ordered categories
- Numerical scaling:
  - Standardization for continuous variables
  - Distribution preservation
- Feature engineering

### 4. Quality Assurance
- Automated validation checks
- Data consistency verification
- Transformation validation
- Output quality control

## Output Files

The analysis generates several professional output files:

1. `output/processed_titanic_data.csv`
   - Clean, transformed dataset
   - Ready for machine learning modeling
   - Properly encoded features

2. `output/analysis_insights.json`
   - Detailed analysis results
   - Data quality metrics
   - Transformation details
   - Processing statistics

3. `output/analysis_report.html`
   - Professional interactive report with Assignment Part section
   - Publication-quality visualizations
   - Comprehensive analysis details
   - Easy to share and view

4. `output/plots/`
   - `data_distribution.png`: Initial data analysis
   - `missing_values.png`: Missing data patterns
   - `correlation_matrix.png`: Feature relationships
   - `transformed_distributions.png`: Final features

5. `titanic_analysis.log`
   - Detailed processing log
   - Error tracking
   - Performance metrics

## Implementation Details

### Project Structure
```
module2/
├── output/
│   ├── plots/
│   │   ├── data_distribution.png
│   │   ├── missing_values.png
│   │   ├── correlation_matrix.png
│   │   └── transformed_distributions.png
│   ├── processed_titanic_data.csv
│   ├── analysis_insights.json
│   └── analysis_report.html
├── Lalit_Nayyar_titanic_analysis.py
├── download_dataset.py
├── requirements.txt
└── README.md
```

### Key Classes and Methods

```python
class TitanicDataAnalyzer:
    def understand_raw_data(self) -> Dict:
        # Analyzes dataset structure and quality
        
    def clean_data(self) -> pd.DataFrame:
        # Handles missing values and feature selection
        
    def transform_data(self) -> pd.DataFrame:
        # Applies feature transformations
        
    def generate_insights(self) -> Dict:
        # Generates comprehensive insights
```

### Processing Statistics

#### 1. Data Understanding
- Total Records: 891
- Total Features: 12
- Numerical Features: 7
- Categorical Features: 5
- Initial Missing Values:
  - Age: 177 records
  - Cabin: 687 records
  - Embarked: 2 records

#### 2. Data Cleaning
- Records After Cleaning: 891
- Features After Cleaning: 9
- Missing Values Handled: 866
- Features Dropped: 3
- Quality Improvements:
  - Age imputation using stratified medians
  - Cabin conversion to binary feature
  - Embarked filled with mode value

#### 3. Feature Engineering
- New Encoded Features: 2
- Scaled Features: 2
- Binary Features: 3
- Transformations:
  - Label encoding for Sex and Embarked
  - Standardization for Age and Fare
  - Binary encoding for Cabin presence

#### 4. Final Output
- Final Records: 891
- Final Features: 9
- Memory Usage: ~68 KB
- Remaining Null Values: 0
- Output Files:
  - Processed dataset (CSV)
  - Analysis insights (JSON)
  - Visual report (HTML)
  - Feature plots (PNG)

## Technical Requirements

### Dependencies
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
seaborn>=0.11.0
matplotlib>=3.4.0
```

### System Requirements
- Python 3.8 or higher
- 4GB RAM recommended
- 100MB free disk space

## Submission Details

### Files Submitted
1. `Lalit_Nayyar_Assignment_2.1.md`
   - Assignment documentation
   - Analysis approach
   - Implementation decisions
   - Results interpretation

2. `Lalit_Nayyar_titanic_analysis.py`
   - Main implementation
   - Data processing pipeline
   - Analysis functions
   - Report generation

3. Output Files
   - Processed dataset
   - Analysis insights
   - Visual report
   - Feature plots

### Contact Information
- **Name:** Lalit Nayyar
- **Email:** lalitnayyar@gmail.com
- **Course:** IIMK Professional Certificate Program
- **Assignment:** Week 2 - Required Assignment 2.1

## License
This project is part of the IIMK Professional Certificate Program coursework and is subject to the program's terms and conditions.
