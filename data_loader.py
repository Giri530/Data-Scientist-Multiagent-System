"""
Data Loader Agent - Handles loading data from various sources
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import requests
from io import StringIO


class DataLoaderAgent:
    """Agent responsible for loading data from various sources"""

    def __init__(self):
        self.supported_formats = ['csv', 'json', 'txt', 'sql', 'api', 'excel']

    def load_data(self, source, source_type='csv', **kwargs):
        """
        Load data from various sources

        Args:
            source: Path to file, URL, or database table name
            source_type: Type of source ('csv', 'json', 'sql', 'api', 'excel')
            **kwargs: Additional parameters for specific loaders

        Returns:
            Dictionary with status, data, and metadata
        """
        try:
            if source_type == 'csv':
                data = self._load_csv(source, **kwargs)
            elif source_type == 'excel':
                data = self._load_excel(source, **kwargs)
            elif source_type == 'json':
                data = self._load_json(source, **kwargs)
            elif source_type == 'sql':
                data = self._load_sql(source, **kwargs)
            elif source_type == 'api':
                data = self._load_api(source, **kwargs)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            return {
                'status': 'success',
                'data': data,
                'info': {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'data': None
            }

    def _load_csv(self, source, **kwargs):
        """Load CSV data from file or URL"""
        if isinstance(source, str) and source.startswith('http'):
            return pd.read_csv(source, **kwargs)
        else:
            return pd.read_csv(source, **kwargs)

    def _load_excel(self, source, **kwargs):
        """Load Excel data from file or URL"""
        if isinstance(source, str) and source.startswith('http'):
            return pd.read_excel(source, **kwargs)
        else:
            return pd.read_excel(source, **kwargs)

    def _load_json(self, source, **kwargs):
        """Load JSON data from file or URL"""
        if isinstance(source, str) and source.startswith('http'):
            response = requests.get(source)
            data = pd.json_normalize(response.json())
        else:
            with open(source, 'r') as f:
                json_data = json.load(f)
            data = pd.json_normalize(json_data)
        return data

    def _load_sql(self, source, **kwargs):
        """Load data from SQL database"""
        database = kwargs.get('database', 'database.db')
        query = kwargs.get('query', f'SELECT * FROM {source}')

        conn = sqlite3.connect(database)
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data

    def _load_api(self, source, **kwargs):
        """Load data from API endpoint"""
        headers = kwargs.get('headers', {})
        params = kwargs.get('params', {})

        response = requests.get(source, headers=headers, params=params)
        response.raise_for_status()

        data = pd.json_normalize(response.json())
        return data

    def get_sample(self, data, n=5):
        """Get a sample of the data for quick inspection"""
        return {
            'head': data.head(n).to_dict('records'),
            'tail': data.tail(n).to_dict('records'),
            'random_sample': data.sample(min(n, len(data))).to_dict('records')
        }
