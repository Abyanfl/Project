'''
=================================================

This program is designed to automate the transformation and loading of data from a CSV file into PostgreSQL, and subsequently into Elasticsearch.  
The dataset used contains information related to a marketing campaign.

The primary goal of this program is to establish a clean and structured data foundation, enabling further analysis using tools such as Kibana for data visualization.  
Ultimately, this supports better decision-making within the context of marketing strategies.
=================================================
'''


import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from elasticsearch import Elasticsearch
import os
import logging

default_args = {
    'owner': 'Abyan',
    'start_date': datetime(2024, 11, 1)
}

with DAG(
    dag_id='DAG',
    description='From CSV to PostgreSQL to Elasticsearch',
    schedule_interval='10-30 9 * * 6',
    default_args=default_args,
    catchup=False
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    @task
    def extract():
        '''
        Function to extract data from CSV and PostgreSQL and save to a file
        '''
        try:
            # PostgreSQL connection
            database = 'airflow'
            username = 'airflow'
            password = 'airflow'
            host = 'postgres'

            postgres_url = f'postgresql+psycopg2://{username}:{password}@{host}/{database}'
            engine = create_engine(postgres_url)
            conn = engine.connect()

            # Read from PostgreSQL (optional, only if needed)
            df_postgres = pd.read_sql('select * from tabel_m3', conn)
            logging.info('Successfully read from PostgreSQL')

            # Read from CSV
            csv_path = '/opt/airflow/data/data_raw.csv'
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at {csv_path}")
            df = pd.read_csv(csv_path)
            logging.info('Successfully read CSV file')

            # Save DataFrame to a temporary file
            output_path = '/opt/airflow/data/data_extract.csv'
            df.to_csv(output_path, index=False)
            if not os.path.exists(output_path):
                raise IOError(f"Failed to save CSV at {output_path}")
            logging.info(f'Data saved to {output_path}')
            print(f'Success extract to {output_path}')

            # Close connections
            conn.close()
            engine.dispose()

            return output_path
        except Exception as e:
            logging.error(f"Extraction failed: {e}", exc_info=True)
            raise

    @task
    def transform(file_path):
        '''
        Function to transform data from the extracted file
        '''
        try:
            # Read the extracted CSV file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Extracted CSV file not found at {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f'Successfully read extracted CSV from {file_path}')

            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Handle missing values
            df['Income'] = df['Income'].replace('', pd.NA)
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df.fillna(df.mode().iloc[0], inplace=True)

            # Convert data types to string for categorical columns
            categorical_columns = ['Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2', 
                                 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response']
            for col in categorical_columns:
                df[col] = df[col].astype(str)

            # Replace numerical codes with meaningful labels
            df['Education'] = df['Education'].replace({
                'Graduation': 'Graduation',
                'PhD': 'PhD',
                'Master': 'Master',
                'Basic': 'Basic',
                '2n Cycle': '2n Cycle'
            })

            df['Marital_Status'] = df['Marital_Status'].replace({
                'Single': 'Single',
                'Together': 'Together',
                'Married': 'Married',
                'Divorced': 'Divorced',
                'Widow': 'Widow',
                'Alone': 'Alone',
                'YOLO': 'Other',
                'Absurd': 'Other'
            })

            # Convert Dt_Customer to datetime and extract year
            df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d')
            df['Year_Customer'] = df['Dt_Customer'].dt.year
            df['Dt_Customer'] = df['Dt_Customer'].astype(str)

            # Round numerical columns
            numerical_columns = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
            for col in numerical_columns:
                df[col] = df[col].round(2)

            # Rename columns to snake_case
            df.rename(columns={
                'ID': 'id',
                'Year_Birth': 'year_birth',
                'Education': 'education',
                'Marital_Status': 'marital_status',
                'Income': 'income',
                'Kidhome': 'kidhome',
                'Teenhome': 'teenhome',
                'Dt_Customer': 'dt_customer',
                'Recency': 'recency',
                'MntWines': 'mnt_wines',
                'MntFruits': 'mnt_fruits',
                'MntMeatProducts': 'mnt_meat_products',
                'MntFishProducts': 'mnt_fish_products',
                'MntSweetProducts': 'mnt_sweet_products',
                'MntGoldProds': 'mnt_gold_prods',
                'NumDealsPurchases': 'num_deals_purchases',
                'NumWebPurchases': 'num_web_purchases',
                'NumCatalogPurchases': 'num_catalog_purchases',
                'NumStorePurchases': 'num_store_purchases',
                'NumWebVisitsMonth': 'num_web_visits_month',
                'AcceptedCmp1': 'accepted_cmp1',
                'AcceptedCmp2': 'accepted_cmp2',
                'AcceptedCmp3': 'accepted_cmp3',
                'AcceptedCmp4': 'accepted_cmp4',
                'AcceptedCmp5': 'accepted_cmp5',
                'Complain': 'complain',
                'Z_CostContact': 'z_cost_contact',
                'Z_Revenue': 'z_revenue',
                'Response': 'response',
                'Year_Customer': 'year_customer'
            }, inplace=True)

            # Save cleaned data
            output_path = '/opt/airflow/data/data_clean.csv'
            df.to_csv(output_path, index=False)
            if not os.path.exists(output_path):
                raise IOError(f"Failed to save cleaned CSV at {output_path}")
            logging.info(f'Preprocessed data saved to {output_path}')
            print('Preprocessed data is success')
            print(df.head())

            return output_path
        except Exception as e:
            logging.error(f"Transformation failed: {e}", exc_info=True)
            raise

    @task
    def loading():
            '''
            Function to load data to ElasticSearch
            '''
            es = Elasticsearch('http://elasticsearch:9200')

            # Read data file
            df = pd.read_csv('/opt/airflow/data/data_clean.csv')

            # Send data to ElasticSearch
            for i, row in df.iterrows():
                res = es.index(index='data', id=i+1, body=row.to_json())
                print(res)

    extract_task = extract()
    transform_task = transform(extract_task)
    loading_task = loading()
    start >> extract_task >> transform_task >> loading_task >> end