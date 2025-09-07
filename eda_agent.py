"""
Exploratory Data Analysis Agent - Handles comprehensive data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EDAAgent:
    """Agent for Exploratory Data Analysis"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_data(self, data, target_column=None):
        """
        Comprehensive EDA analysis
        Args:
            data: Input DataFrame
            target_column: Optional target variable for supervised analysis
        Returns:
            Dictionary containing comprehensive analysis results
        """
        analysis = {}

        # Basic statistics
        analysis['basic_stats'] = self._basic_statistics(data)

        # Correlation analysis
        analysis['correlations'] = self._correlation_analysis(data)

        # Distribution analysis
        analysis['distributions'] = self._distribution_analysis(data)

        # Feature insights
        analysis['feature_insights'] = self._feature_insights(data)

        # Target analysis (if target column provided)
        if target_column and target_column in data.columns:
            analysis['target_analysis'] = self._target_analysis(data, target_column)

        # Data quality insights
        analysis['data_quality'] = self._data_quality_insights(data)

        return {
            'status': 'success',
            'analysis': analysis,
            'visualization_recommendations': self._get_visualization_recommendations(data)
        }

    def _basic_statistics(self, data):
        """Generate comprehensive statistical summary"""
        stats = {}

        # Overall info
        stats['shape'] = data.shape
        stats['dtypes'] = data.dtypes.to_dict()
        stats['memory_usage'] = f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"

        # Numeric summary
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            desc = numeric_data.describe()
            stats['numeric_summary'] = desc.to_dict()

            # Additional statistics
            stats['numeric_extended'] = {}
            for col in numeric_data.columns:
                stats['numeric_extended'][col] = {
                    'variance': numeric_data[col].var(),
                    'skewness': numeric_data[col].skew(),
                    'kurtosis': numeric_data[col].kurtosis(),
                    'coefficient_of_variation': numeric_data[col].std() / numeric_data[col].mean() if numeric_data[col].mean() != 0 else np.inf
                }

        # Categorical summary
        categorical_data = data.select_dtypes(include=['object', 'category'])
        if not categorical_data.empty:
            stats['categorical_summary'] = {}
            for col in categorical_data.columns:
                stats['categorical_summary'][col] = {
                    'unique_count': categorical_data[col].nunique(),
                    'most_frequent': categorical_data[col].mode().iloc[0] if len(categorical_data[col].mode()) > 0 else None,
                    'frequency_of_most_frequent': categorical_data[col].value_counts().iloc[0] if len(categorical_data[col]) > 0 else 0
                }

        # Missing values
        stats['missing_values'] = data.isnull().sum().to_dict()

        # Unique values count
        stats['unique_values'] = {col: data[col].nunique() for col in data.columns}

        return stats

    def _correlation_analysis(self, data):
        """Analyze correlations between numeric variables"""
        numeric_data = data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}

        # Correlation matrix
        corr_matrix = numeric_data.corr()

        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'very_strong' if abs(corr_val) > 0.9 else 'strong'
                    })

        # Find moderate correlations
        moderate_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and 0.3 <= abs(corr_val) <= 0.7:
                    moderate_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_corr,
            'moderate_correlations': moderate_corr[:10],  # Limit to top 10
            'summary': {
                'total_pairs': len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2,
                'strong_correlations_count': len(strong_corr),
                'moderate_correlations_count': len(moderate_corr)
            }
        }

    def _distribution_analysis(self, data):
        """Analyze distributions of all variables"""
        distributions = {}

        for col in data.columns:
            col_info = {'column': col, 'dtype': str(data[col].dtype)}

            if data[col].dtype in ['object', 'category']:
                # Categorical distribution
                value_counts = data[col].value_counts()
                col_info.update({
                    'type': 'categorical',
                    'unique_count': len(value_counts),
                    'top_values': value_counts.head(10).to_dict(),
                    'entropy': stats.entropy(value_counts.values) if len(value_counts) > 1 else 0,
                    'most_frequent_percentage': (value_counts.iloc[0] / len(data)) * 100 if len(value_counts) > 0 else 0
                })
            else:
                # Numerical distribution
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    col_info.update({
                        'type': 'numerical',
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'outliers_iqr': self._count_outliers_iqr(col_data),
                        'normality_test': self._test_normality(col_data)
                    })

            distributions[col] = col_info

        return distributions

    def _feature_insights(self, data):
        """Generate feature insights and recommendations"""
        insights = []

        # Identify potential target variables
        for col in data.columns:
            unique_count = data[col].nunique()
            if unique_count == 2:
                insights.append({
                    'type': 'potential_target',
                    'feature': col,
                    'insight': f'{col} is binary - potential target for classification'
                })
            elif unique_count < 10 and data[col].dtype in ['object', 'string']:
                insights.append({
                    'type': 'low_cardinality',
                    'feature': col,
                    'insight': f'{col} has low cardinality ({unique_count}) - good for classification target'
                })

        # Identify high cardinality categorical features
        for col in data.select_dtypes(include=['object']).columns:
            unique_count = data[col].nunique()
            if unique_count > 50:
                insights.append({
                    'type': 'high_cardinality',
                    'feature': col,
                    'insight': f'{col} has high cardinality ({unique_count}) - consider target encoding or grouping'
                })

        # Identify constant or near-constant features
        for col in data.columns:
            unique_count = data[col].nunique()
            if unique_count == 1:
                insights.append({
                    'type': 'constant_feature',
                    'feature': col,
                    'insight': f'{col} is constant - consider removing'
                })
            elif unique_count / len(data) < 0.01:
                insights.append({
                    'type': 'near_constant',
                    'feature': col,
                    'insight': f'{col} is near-constant ({unique_count} unique values) - low information content'
                })

        # Identify features with many missing values
        missing_threshold = 0.5
        for col in data.columns:
            missing_pct = data[col].isnull().sum() / len(data)
            if missing_pct > missing_threshold:
                insights.append({
                    'type': 'high_missing',
                    'feature': col,
                    'insight': f'{col} has {missing_pct:.1%} missing values - consider imputation or removal'
                })

        return insights

    def _target_analysis(self, data, target_column):
        """Analyze target variable and its relationships"""
        target = data[target_column]
        analysis = {}

        # Target distribution
        if target.dtype in ['object', 'category']:
            # Classification target
            value_counts = target.value_counts()
            analysis['type'] = 'classification'
            analysis['classes'] = value_counts.to_dict()
            analysis['class_balance'] = {
                'balanced': max(value_counts) / min(value_counts) < 3,
                'ratio': max(value_counts) / min(value_counts)
            }
        else:
            # Regression target
            analysis['type'] = 'regression'
            analysis['distribution'] = {
                'mean': target.mean(),
                'median': target.median(),
                'std': target.std(),
                'skewness': target.skew(),
                'kurtosis': target.kurtosis()
            }

        # Feature-target relationships
        feature_relationships = []
        other_features = [col for col in data.columns if col != target_column]

        for feature in other_features[:20]:  # Limit to first 20 features
            if data[feature].dtype in [np.number]:
                if analysis['type'] == 'classification':
                    # ANOVA F-test for numeric feature vs categorical target
                    try:
                        groups = [data[data[target_column] == cls][feature].dropna()
                                for cls in target.unique()]
                        f_stat, p_val = stats.f_oneway(*groups)
                        feature_relationships.append({
                            'feature': feature,
                            'test': 'ANOVA',
                            'f_statistic': f_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        })
                    except:
                        pass
                else:
                    # Correlation for numeric feature vs numeric target
                    corr, p_val = stats.pearsonr(data[feature].dropna(),
                                                target[data[feature].notna()])
                    feature_relationships.append({
                        'feature': feature,
                        'test': 'Correlation',
                        'correlation': corr,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    })

        analysis['feature_relationships'] = feature_relationships

        return analysis

    def _data_quality_insights(self, data):
        """Generate data quality insights"""
        insights = []

        # Overall data quality score
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        quality_score = (total_cells - missing_cells) / total_cells

        insights.append({
            'type': 'overall_quality',
            'score': quality_score,
            'interpretation': 'excellent' if quality_score > 0.95 else
                            'good' if quality_score > 0.85 else
                            'fair' if quality_score > 0.7 else 'poor'
        })

        # Duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            insights.append({
                'type': 'duplicates',
                'count': duplicate_count,
                'percentage': (duplicate_count / len(data)) * 100
            })

        return insights

    def _count_outliers_iqr(self, series):
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)

    def _test_normality(self, series, max_samples=5000):
        """Test normality using Shapiro-Wilk test"""
        try:
            if len(series) > max_samples:
                series_sample = series.sample(max_samples)
            else:
                series_sample = series

            stat, p_value = stats.shapiro(series_sample)
            return {
                'test_statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {'test_statistic': None, 'p_value': None, 'is_normal': None}

    def _get_visualization_recommendations(self, data):
        """Generate visualization recommendations based on data characteristics"""
        recommendations = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # Distribution plots
        if len(numeric_cols) > 0:
            recommendations.append({
                'type': 'histogram',
                'purpose': 'Show distribution of numeric variables',
                'columns': list(numeric_cols[:5])
            })

            recommendations.append({
                'type': 'box_plot',
                'purpose': 'Identify outliers in numeric variables',
                'columns': list(numeric_cols[:5])
            })

        # Categorical plots
        if len(categorical_cols) > 0:
            recommendations.append({
                'type': 'bar_chart',
                'purpose': 'Show frequency of categorical variables',
                'columns': list(categorical_cols[:5])
            })

        # Relationship plots
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'correlation_heatmap',
                'purpose': 'Show correlations between numeric variables',
                'columns': list(numeric_cols)
            })

            recommendations.append({
                'type': 'scatter_plot',
                'purpose': 'Show relationships between numeric variables',
                'columns': list(numeric_cols[:4])
            })

        # Mixed plots
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            recommendations.append({
                'type': 'grouped_box_plot',
                'purpose': 'Show numeric distributions by categorical groups',
                'numeric_columns': list(numeric_cols[:3]),
                'categorical_columns': list(categorical_cols[:2])
            })

        return recommendations

    def generate_insights_summary(self, analysis_results):
        """Generate a human-readable summary of key insights"""
        if analysis_results['status'] != 'success':
            return "Analysis failed"

        analysis = analysis_results['analysis']
        insights = []

        # Basic stats insights
        basic_stats = analysis['basic_stats']
        insights.append(f"Dataset contains {basic_stats['shape'][0]:,} rows and {basic_stats['shape'][1]} columns")

        # Missing values insight
        missing_total = sum(basic_stats['missing_values'].values())
        if missing_total > 0:
            insights.append(f"Found {missing_total:,} missing values across the dataset")

        # Correlation insights
        if 'correlations' in analysis and 'strong_correlations' in analysis['correlations']:
            strong_corr_count = len(analysis['correlations']['strong_correlations'])
            if strong_corr_count > 0:
                insights.append(f"Identified {strong_corr_count} strong correlations between variables")

        # Feature insights
        if 'feature_insights' in analysis:
            feature_insights = analysis['feature_insights']
            potential_targets = [i for i in feature_insights if i['type'] == 'potential_target']
            if potential_targets:
                insights.append(f"Found {len(potential_targets)} potential target variables for machine learning")

        return insights
