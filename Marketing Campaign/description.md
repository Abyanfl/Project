# Marketing Campaign 

This project implements an automated **ETL (Extract, Transform, Load)** pipeline using **Apache Airflow** to process a marketing campaign dataset. The pipeline extracts data from a CSV file, transforms it into a clean and structured format, loads it into a **PostgreSQL** database, and subsequently transfers it to **Elasticsearch** for advanced analytics and visualization.

## Project Overview
The pipeline processes a dataset containing customer demographics and campaign response data, such as education, marital status, income, and purchase behavior. The goal is to create a robust data foundation for analysis, enabling visualization in tools like **Kibana** to support data-driven marketing decisions.

### Key Features
- **Extraction**: Reads raw data from a CSV file (`P2M3_abyan_data_raw.csv`) and optionally from a PostgreSQL table.
- **Transformation**: Cleans the dataset by handling missing values, removing duplicates, converting data types, standardizing categorical values, and renaming columns to snake_case.
- **Loading**: Stores transformed data in PostgreSQL and indexes it in Elasticsearch for efficient querying and visualization.
- **Automation**: Uses Airflow to schedule and orchestrate the ETL process, running weekly at 09:10-09:30 on Saturdays.

### Dataset
The dataset includes customer attributes (e.g., `Year_Birth`, `Education`, `Income`) and campaign metrics (e.g., `AcceptedCmp1`, `Response`). Transformations ensure consistent data types, meaningful categorical labels, and rounded numerical values for analysis.

### Technologies
- **Apache Airflow**: Workflow orchestration
- **Pandas**: Data processing and transformation
- **PostgreSQL**: Relational database storage
- **Elasticsearch**: Search and analytics engine
- **Python**: Core programming language

### Purpose
This pipeline enables marketers and analysts to access clean, structured data for insights into customer behavior and campaign performance, facilitating better decision-making through visualizations in Kibana.
