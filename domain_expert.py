"""
Domain Expert Agent - Provides domain-specific insights and recommendations
"""

import pandas as pd
import numpy as np
import re


class DomainExpertAgent:
    """Agent that provides domain-specific insights and recommendations"""

    def __init__(self):
        self.domain_knowledge = {
            'finance': {
                'key_metrics': ['roi', 'profit', 'revenue', 'cost', 'price', 'amount', 'balance',
                              'rate', 'interest', 'yield', 'return', 'income', 'expense'],
                'common_features': ['account', 'transaction', 'customer_id', 'date', 'currency',
                                  'credit', 'debit', 'portfolio', 'risk'],
                'insights': [
                    'Look for seasonal patterns in financial data',
                    'Check for outliers in transaction amounts',
                    'Consider risk-adjusted metrics for portfolio analysis',
                    'Time-based features are crucial for financial modeling'
                ],
                'feature_engineering': [
                    'Create rolling averages for financial metrics',
                    'Extract time-based features (month, quarter, year)',
                    'Calculate ratios between financial metrics',
                    'Create lagged features for time series analysis'
                ]
            },
            'healthcare': {
                'key_metrics': ['age', 'bmi', 'weight', 'height', 'blood_pressure', 'heart_rate',
                              'diagnosis', 'treatment', 'dose', 'duration'],
                'common_features': ['patient_id', 'doctor', 'hospital', 'medication', 'symptoms',
                                  'medical_history', 'lab_results'],
                'insights': [
                    'Age correlation is important in healthcare analysis',
                    'Consider demographic factors (gender, ethnicity)',
                    'Look for comorbidities and drug interactions',
                    'Temporal patterns in symptoms and treatments matter'
                ],
                'feature_engineering': [
                    'Create BMI categories (underweight, normal, overweight, obese)',
                    'Calculate age groups or bins',
                    'Create interaction features between symptoms',
                    'Encode medical history as binary features'
                ]
            },
            'retail': {
                'key_metrics': ['sales', 'price', 'quantity', 'revenue', 'profit', 'discount',
                              'margin', 'units_sold', 'inventory'],
                'common_features': ['product', 'category', 'brand', 'customer_id', 'store',
                                  'seasonality', 'promotion', 'location'],
                'insights': [
                    'Check for seasonal trends in sales data',
                    'Customer segmentation opportunities exist',
                    'Price elasticity analysis is valuable',
                    'Geographic patterns in sales performance'
                ],
                'feature_engineering': [
                    'Create customer lifetime value metrics',
                    'Calculate recency, frequency, monetary (RFM) features',
                    'Extract seasonal indicators',
                    'Create product affinity features'
                ]
            },
            'marketing': {
                'key_metrics': ['ctr', 'conversion_rate', 'cpa', 'roas', 'impressions', 'clicks',
                              'bounce_rate', 'engagement', 'reach'],
                'common_features': ['campaign', 'channel', 'audience', 'creative', 'budget',
                                  'demographics', 'device', 'location'],
                'insights': [
                    'Multi-touch attribution is complex',
                    'Seasonality affects campaign performance',
                    'Audience segmentation drives performance',
                    'Cross-channel interactions are important'
                ],
                'feature_engineering': [
                    'Create funnel conversion features',
                    'Calculate attribution weights',
                    'Extract time-since-last-interaction features',
                    'Create audience overlap indicators'
                ]
            },
            'manufacturing': {
                'key_metrics': ['temperature', 'pressure', 'speed', 'quality', 'defect_rate',
                              'efficiency', 'downtime', 'throughput'],
                'common_features': ['machine', 'operator', 'shift', 'material', 'batch',
                                  'sensor_reading', 'maintenance'],
                'insights': [
                    'Equipment maintenance schedules affect quality',
                    'Environmental conditions impact production',
                    'Operator experience correlates with quality',
                    'Supply chain disruptions affect throughput'
                ],
                'feature_engineering': [
                    'Create rolling statistics for sensor data',
                    'Calculate time-since-maintenance features',
                    'Create shift and time-based features',
                    'Extract statistical process control features'
                ]
            }
        }

    def provide_domain_insights(self, data, domain=None, target_column=None):
        """
        Provide comprehensive domain-specific insights
        Args:
            data: Input DataFrame
            domain: Specific domain (optional, will auto-detect if None)
            target_column: Target variable for supervised learning
        Returns:
            Dictionary with domain insights and recommendations
        """
        if not domain:
            domain = self._detect_domain(data)

        insights = []
        recommendations = []
        feature_engineering_suggestions = []

        # Domain-specific analysis
        if domain in self.domain_knowledge:
            domain_info = self.domain_knowledge[domain]

            # Check for domain-relevant columns
            relevant_features = self._find_relevant_features(data, domain_info)
            if relevant_features:
                insights.append(f"Found {len(relevant_features)} domain-relevant features: {relevant_features}")

            # Add domain-specific recommendations
            recommendations.extend(domain_info['insights'])
            feature_engineering_suggestions.extend(domain_info['feature_engineering'])

        # Generic insights based on data characteristics
        generic_insights = self._generate_generic_insights(data)
        insights.extend(generic_insights)

        # Target-specific recommendations
        if target_column and target_column in data.columns:
            target_insights = self._analyze_target_for_domain(data, target_column, domain)
            insights.extend(target_insights)

        # Data size recommendations
        size_recommendations = self._get_size_recommendations(data)
        recommendations.extend(size_recommendations)

        # Feature engineering suggestions based on actual data
        data_based_fe = self._suggest_data_based_feature_engineering(data)
        feature_engineering_suggestions.extend(data_based_fe)

        return {
            'detected_domain': domain,
            'confidence': self._calculate_domain_confidence(data, domain),
            'insights': insights,
            'recommendations': recommendations,
            'feature_engineering_suggestions': feature_engineering_suggestions,
            'modeling_recommendations': self._get_modeling_recommendations(data, domain, target_column)
        }

    def _detect_domain(self, data):
        """Detect domain based on column names and patterns"""
        column_text = ' '.join(data.columns).lower()

        domain_scores = {}
        for domain, info in self.domain_knowledge.items():
            score = 0
            all_keywords = info['key_metrics'] + info['common_features']

            for keyword in all_keywords:
                # Exact match
                if keyword in column_text:
                    score += 2
                # Partial match
                elif any(keyword in col for col in column_text.split()):
                    score += 1

            domain_scores[domain] = score

        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'

    def _calculate_domain_confidence(self, data, domain):
        """Calculate confidence score for domain detection"""
        if domain == 'general':
            return 0.0

        if domain not in self.domain_knowledge:
            return 0.0

        column_text = ' '.join(data.columns).lower()
        domain_info = self.domain_knowledge[domain]
        all_keywords = domain_info['key_metrics'] + domain_info['common_features']

        matches = sum(1 for keyword in all_keywords if keyword in column_text)
        confidence = min(matches / 5, 1.0)  # Normalize to max 1.0

        return confidence

    def _find_relevant_features(self, data, domain_info):
        """Find features relevant to the domain"""
        relevant_features = []
        column_names = [col.lower() for col in data.columns]

        all_keywords = domain_info['key_metrics'] + domain_info['common_features']

        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in all_keywords):
                relevant_features.append(col)

        return relevant_features

    def _generate_generic_insights(self, data):
        """Generate insights based on general data characteristics"""
        insights = []

        # High-dimensional data
        if len(data.columns) > 50:
            insights.append("High-dimensional dataset - consider dimensionality reduction techniques")

        # Wide vs tall data
        if data.shape[1] > data.shape[0]:
            insights.append("Wide dataset (more features than samples) - risk of overfitting")

        # Mixed data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            insights.append("Mixed data types detected - consider different preprocessing for numeric vs categorical")

        # High cardinality features
        high_card_features = []
        for col in categorical_cols:
            if data[col].nunique() > 20:
                high_card_features.append(col)

        if high_card_features:
            insights.append(f"High cardinality categorical features detected: {high_card_features}")

        # Imbalanced features
        for col in categorical_cols:
            if data[col].value_counts().iloc[0] / len(data) > 0.95:
                insights.append(f"Feature '{col}' is highly imbalanced (>95% single value)")

        return insights

    def _analyze_target_for_domain(self, data, target_column, domain):
        """Analyze target variable in domain context"""
        insights = []
        target = data[target_column]

        # Classification vs Regression
        if target.dtype in ['object', 'category'] or target.nunique() < 20:
            problem_type = 'classification'
            class_counts = target.value_counts()

            if len(class_counts) == 2:
                insights.append("Binary classification problem detected")
                # Check for class imbalance
                ratio = class_counts.iloc[0] / class_counts.iloc[1]
                if ratio > 3:
                    insights.append(f"Class imbalance detected (ratio: {ratio:.1f}:1)")
            else:
                insights.append(f"Multi-class classification with {len(class_counts)} classes")
        else:
            problem_type = 'regression'
            insights.append("Regression problem detected")

            # Check for skewed target
            if abs(target.skew()) > 1:
                insights.append("Target variable is skewed - consider transformation")

        # Domain-specific target analysis
        if domain in self.domain_knowledge:
            domain_info = self.domain_knowledge[domain]

            if domain == 'finance' and problem_type == 'regression':
                insights.append("Consider log transformation for financial targets")
            elif domain == 'healthcare' and problem_type == 'classification':
                insights.append("Medical diagnosis prediction - ensure proper validation strategy")
            elif domain == 'retail' and 'sales' in target_column.lower():
                insights.append("Sales prediction - consider seasonal effects")

        return insights

    def _get_size_recommendations(self, data):
        """Get recommendations based on dataset size"""
        recommendations = []
        n_rows, n_cols = data.shape

        if n_rows < 1000:
            recommendations.append("Small dataset - use cross-validation and simple models")
        elif n_rows > 100000:
            recommendations.append("Large dataset - consider sampling for initial exploration")

        if n_cols > 100:
            recommendations.append("Many features - consider feature selection techniques")

        if n_rows < n_cols:
            recommendations.append("More features than samples - high risk of overfitting")

        return recommendations

    def _suggest_data_based_feature_engineering(self, data):
        """Suggest feature engineering based on actual data"""
        suggestions = []

        # Date columns
        date_cols = []
        for col in data.columns:
            if data[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                date_cols.append(col)

        if date_cols:
            suggestions.append(f"Extract temporal features from date columns: {date_cols}")

        # Text columns that might need processing
        text_cols = []
        for col in data.select_dtypes(include=['object']).columns:
            # Check if contains long text
            avg_length = data[col].astype(str).str.len().mean()
            if avg_length > 20:
                text_cols.append(col)

        if text_cols:
            suggestions.append(f"Text columns may need NLP preprocessing: {text_cols}")

        # Numeric columns for interactions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            suggestions.append("Consider creating interaction features between numeric variables")
            suggestions.append("Create polynomial features for non-linear relationships")

        # Categorical columns for encoding
        categorical_cols = data.select_dtypes(include=['object']).columns
        low_card_cols = [col for col in categorical_cols if data[col].nunique() <= 10]
        high_card_cols = [col for col in categorical_cols if data[col].nunique() > 10]

        if low_card_cols:
            suggestions.append(f"One-hot encode low cardinality features: {low_card_cols}")

        if high_card_cols:
            suggestions.append(f"Consider target encoding for high cardinality features: {high_card_cols}")

        return suggestions

    def _get_modeling_recommendations(self, data, domain, target_column):
        """Get modeling recommendations based on domain and data characteristics"""
        recommendations = []

        n_rows, n_cols = data.shape

        # Based on data size
        if n_rows < 1000:
            recommendations.extend([
                "Use simpler models (Linear Regression, Decision Trees)",
                "Implement robust cross-validation",
                "Avoid complex ensemble methods"
            ])
        elif n_rows > 10000:
            recommendations.extend([
                "Can use complex models (Random Forest, Gradient Boosting)",
                "Deep learning models are viable",
                "Consider ensemble methods"
            ])

        # Based on domain
        if domain == 'finance':
            recommendations.extend([
                "Consider time series models if temporal data is present",
                "Use robust models that handle outliers well",
                "Implement proper risk management in model validation"
            ])
        elif domain == 'healthcare':
            recommendations.extend([
                "Ensure model interpretability for medical decisions",
                "Use stratified sampling for validation",
                "Consider regulatory compliance requirements"
            ])
        elif domain == 'retail':
            recommendations.extend([
                "Account for seasonality in modeling",
                "Consider customer segmentation approaches",
                "Use models that can handle promotional effects"
            ])

        # Based on target type
        if target_column and target_column in data.columns:
            target = data[target_column]

            if target.dtype in ['object', 'category']:
                # Classification
                recommendations.append("Classification problem - consider precision/recall trade-offs")

                if target.nunique() == 2:
                    recommendations.append("Binary classification - ROC-AUC is a good metric")
                else:
                    recommendations.append("Multi-class classification - use macro/micro averaged metrics")
            else:
                # Regression
                recommendations.append("Regression problem - focus on RMSE and MAE metrics")

                if target.min() >= 0:
                    recommendations.append("Non-negative target - consider specialized loss functions")

        return recommendations
