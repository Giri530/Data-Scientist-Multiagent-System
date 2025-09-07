"""
Data Cleaning Agent - Handles data preprocessing and cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataCleaningAgent:
    """Agent responsible for data cleaning and preprocessing"""

    def __init__(self):
        self.cleaning_report = {}
        self.label_encoders = {}

    def clean_data(self, data, aggressive_cleaning=False):
        """
        Comprehensive data cleaning
        Args:
            data: Input DataFrame
            aggressive_cleaning: Whether to apply more aggressive cleaning
        Returns:
            Dictionary with cleaned data and cleaning report
        """
        cleaned_data = data.copy()
        report = {
            'original_shape': data.shape,
            'cleaning_steps': []
        }

        # Handle missing values
        missing_info = self._handle_missing_values(cleaned_data)
        report['missing_values'] = missing_info
        report['cleaning_steps'].append('Missing values handled')

        # Remove duplicates
        duplicates_removed = self._remove_duplicates(cleaned_data)
        report['duplicates_removed'] = duplicates_removed
        if duplicates_removed > 0:
            report['cleaning_steps'].append(f'Removed {duplicates_removed} duplicates')

        # Handle outliers
        if aggressive_cleaning:
            outliers_info = self._handle_outliers(cleaned_data)
            report['outliers'] = outliers_info
            report['cleaning_steps'].append('Outliers handled')

        # Data type optimization
        type_changes = self._optimize_dtypes(cleaned_data)
        report['type_changes'] = type_changes
        if type_changes:
            report['cleaning_steps'].append('Data types optimized')

        # Handle infinite values
        inf_handled = self._handle_infinite_values(cleaned_data)
        if inf_handled:
            report['cleaning_steps'].append('Infinite values handled')

        report['final_shape'] = cleaned_data.shape
        report['rows_removed'] = data.shape[0] - cleaned_data.shape[0]

        return {
            'status': 'success',
            'data': cleaned_data,
            'cleaning_report': report
        }

    def _handle_missing_values(self, data, strategy='smart'):
        """Handle missing values based on column type and distribution"""
        missing_info = {}

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(data)) * 100
                }

                if data[col].dtype in ['object', 'string']:
                    # Fill with mode for categorical
                    mode_val = data[col].mode()
                    if len(mode_val) > 0:
                        data[col].fillna(mode_val[0], inplace=True)
                        missing_info[col]['strategy'] = f'filled_with_mode: {mode_val[0]}'
                    else:
                        data[col].fillna('Unknown', inplace=True)
                        missing_info[col]['strategy'] = 'filled_with_unknown'
                else:
                    # For numerical columns, choose between mean/median based on skewness
                    skewness = abs(data[col].skew())
                    if skewness > 1:  # Highly skewed, use median
                        fill_value = data[col].median()
                        data[col].fillna(fill_value, inplace=True)
                        missing_info[col]['strategy'] = f'filled_with_median: {fill_value}'
                    else:  # Relatively normal, use mean
                        fill_value = data[col].mean()
                        data[col].fillna(fill_value, inplace=True)
                        missing_info[col]['strategy'] = f'filled_with_mean: {fill_value}'

        return missing_info

    def _remove_duplicates(self, data):
        """Remove duplicate rows"""
        initial_count = len(data)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return initial_count - len(data)

    def _handle_outliers(self, data, method='iqr'):
        """Handle outliers using IQR method"""
        outlier_info = {}

        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:  # Skip columns with no variance
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

                # Cap outliers instead of removing (more conservative)
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound

        return outlier_info

    def _optimize_dtypes(self, data):
        """Optimize data types for memory efficiency"""
        type_changes = {}

        for col in data.columns:
            original_type = str(data[col].dtype)

            # Try to convert object columns to numeric
            if data[col].dtype == 'object':
                try:
                    # First try to convert to numeric
                    numeric_series = pd.to_numeric(data[col], errors='coerce')
                    if not numeric_series.isnull().all():
                        data[col] = numeric_series
                        type_changes[col] = f"{original_type} -> {data[col].dtype}"
                        continue
                except:
                    pass

                # Try to convert to datetime
                try:
                    datetime_series = pd.to_datetime(data[col], errors='coerce')
                    if not datetime_series.isnull().all():
                        data[col] = datetime_series
                        type_changes[col] = f"{original_type} -> datetime64[ns]"
                        continue
                except:
                    pass

            # Optimize integer types
            elif data[col].dtype in ['int64']:
                if data[col].min() >= 0:
                    if data[col].max() <= 255:
                        data[col] = data[col].astype('uint8')
                        type_changes[col] = f"{original_type} -> uint8"
                    elif data[col].max() <= 65535:
                        data[col] = data[col].astype('uint16')
                        type_changes[col] = f"{original_type} -> uint16"
                    elif data[col].max() <= 4294967295:
                        data[col] = data[col].astype('uint32')
                        type_changes[col] = f"{original_type} -> uint32"
                else:
                    if data[col].min() >= -128 and data[col].max() <= 127:
                        data[col] = data[col].astype('int8')
                        type_changes[col] = f"{original_type} -> int8"
                    elif data[col].min() >= -32768 and data[col].max() <= 32767:
                        data[col] = data[col].astype('int16')
                        type_changes[col] = f"{original_type} -> int16"
                    elif data[col].min() >= -2147483648 and data[col].max() <= 2147483647:
                        data[col] = data[col].astype('int32')
                        type_changes[col] = f"{original_type} -> int32"

            # Optimize float types
            elif data[col].dtype in ['float64']:
                if data[col].min() >= np.finfo(np.float32).min and data[col].max() <= np.finfo(np.float32).max:
                    data[col] = data[col].astype('float32')
                    type_changes[col] = f"{original_type} -> float32"

        return type_changes

    def _handle_infinite_values(self, data):
        """Handle infinite values in the dataset"""
        inf_cols = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if np.isinf(data[col]).any():
                inf_cols.append(col)
                # Replace infinite values with NaN, then fill with column median
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                data[col].fillna(data[col].median(), inplace=True)

        return len(inf_cols) > 0

    def get_data_quality_report(self, data):
        """Generate a comprehensive data quality report"""
        report = {}

        # Basic info
        report['shape'] = data.shape
        report['dtypes'] = data.dtypes.to_dict()

        # Missing values
        missing = data.isnull().sum()
        report['missing_values'] = {
            'total': missing.sum(),
            'by_column': missing[missing > 0].to_dict(),
            'percentage': (missing / len(data) * 100)[missing > 0].to_dict()
        }

        # Duplicates
        report['duplicates'] = data.duplicated().sum()

        # Unique values
        report['unique_values'] = {col: data[col].nunique() for col in data.columns}

        # Memory usage
        report['memory_usage'] = {
            'total_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'by_column': (data.memory_usage(deep=True) / 1024**2).to_dict()
        }

        return report
