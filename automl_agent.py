"""
AutoML Agent - Advanced automated machine learning with hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class AutoMLAgent:
    """Advanced AutoML agent with hyperparameter optimization"""

    def __init__(self):
        self.best_models = {}
        self.optimization_results = {}
        self.search_spaces = {}

    def auto_optimize(self, data, target_column, problem_type=None, time_budget=300, optimization_metric=None):
        """
        Automated model selection and hyperparameter optimization
        Args:
            data: Input DataFrame
            target_column: Target variable name
            problem_type: 'classification' or 'regression' (auto-detect if None)
            time_budget: Time budget in seconds
            optimization_metric: Metric to optimize (auto-select if None)
        Returns:
            Dictionary with optimization results
        """
        print(f"🔧 Starting AutoML optimization (Time budget: {time_budget}s)...")

        try:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]

            if problem_type is None:
                problem_type = self._detect_problem_type(y)

            print(f"📊 Detected problem type: {problem_type}")

            # Preprocess
            X_processed = self._preprocess_features(X)

            # Encode target
            if 'classification' in problem_type and y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y.copy()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Select optimization metric
            if optimization_metric is None:
                optimization_metric = self._select_optimization_metric(problem_type)

            # Define models and parameter grids
            models_params = self._get_models_with_params(problem_type)

            best_score = -np.inf if 'classification' in problem_type else np.inf
            best_model_info = None

            # Optimize each model
            for model_name, (model, param_grid) in models_params.items():
                print(f"🔍 Optimizing {model_name}...")

                try:
                    # Use RandomizedSearchCV for efficiency
                    search = RandomizedSearchCV(
                        model, param_grid,
                        cv=5,
                        scoring=optimization_metric,
                        n_iter=min(50, len(param_grid) * 10),  # Adaptive iterations
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )

                    search.fit(X_train_scaled, y_train)

                    # Evaluate on test set
                    y_pred = search.best_estimator_.predict(X_test_scaled)

                    if 'classification' in problem_type:
                        test_score = accuracy_score(y_test, y_pred)
                        is_better = test_score > best_score
                        additional_metrics = {
                            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        }
                    else:
                        test_score = mean_squared_error(y_test, y_pred, squared=False)
                        is_better = test_score < best_score
                        additional_metrics = {
                            'mae': np.mean(np.abs(y_test - y_pred))
                        }

                    if is_better:
                        best_score = test_score
                        best_model_info = {
                            'name': model_name,
                            'model': search.best_estimator_,
                            'score': test_score,
                            'best_params': search.best_params_,
                            'cv_score': search.best_score_,
                            'predictions': y_pred,
                            **additional_metrics
                        }

                    self.optimization_results[model_name] = {
                        'best_params': search.best_params_,
                        'cv_score': search.best_score_,
                        'test_score': test_score,
                        'param_grid_size': len(param_grid),
                        'iterations_performed': search.n_splits_ * min(50, len(param_grid) * 10),
                        **additional_metrics
                    }

                except Exception as e:
                    print(f"❌ Error optimizing {model_name}: {e}")
                    self.optimization_results[model_name] = {'error': str(e)}

            # Generate optimization insights
            optimization_insights = self._generate_optimization_insights(self.optimization_results, problem_type)

            return {
                'status': 'success',
                'best_model': best_model_info,
                'all_results': self.optimization_results,
                'problem_type': problem_type,
                'optimization_metric': optimization_metric,
                'insights': optimization_insights,
                'preprocessing_info': {
                    'features_processed': X_processed.shape[1],
                    'original_features': X.shape[1],
                    'scaler_used': 'StandardScaler'
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'details': 'AutoML optimization failed'
            }

    def _detect_problem_type(self, target):
        """Detect problem type"""
        unique_count = target.nunique()

        if target.dtype == 'object':
            return 'classification'
        elif unique_count == 2:
            return 'binary_classification'
        elif unique_count < 20:
            return 'multiclass_classification'
        else:
            return 'regression'

    def _preprocess_features(self, X):
        """Preprocess features for optimization"""
        X_processed = X.copy()

        # Handle categorical variables
        for col in X_processed.select_dtypes(include=['object']).columns:
            if X_processed[col].nunique() <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                X_processed = pd.concat([X_processed, dummies], axis=1)
                X_processed.drop(columns=[col], inplace=True)
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())

        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())

        return X_processed

    def _select_optimization_metric(self, problem_type):
        """Select appropriate optimization metric based on problem type"""
        if problem_type == 'binary_classification':
            return 'roc_auc'
        elif 'classification' in problem_type:
            return 'accuracy'
        else:
            return 'neg_mean_squared_error'

    def _get_models_with_params(self, problem_type):
        """Get models with parameter grids for optimization"""
        if 'classification' in problem_type:
            return {
                'Random Forest': (
                    RandomForestClassifier(random_state=42),
                    {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                ),
                'SVM': (
                    SVC(random_state=42, probability=True),
                    {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['rbf', 'linear', 'poly'],
                        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                    }
                ),
                'Logistic Regression': (
                    LogisticRegression(random_state=42, max_iter=1000),
                    {
                        'C': [0.01, 0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2', 'elasticnet'],
                        'solver': ['liblinear', 'saga', 'lbfgs'],
                        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only for elasticnet
                    }
                )
            }
        else:
            return {
                'Random Forest': (
                    RandomForestRegressor(random_state=42),
                    {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                ),
                'SVR': (
                    SVR(),
                    {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['rbf', 'linear', 'poly'],
                        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                        'epsilon': [0.01, 0.1, 0.2, 0.5]
                    }
                ),
                'Ridge': (
                    Ridge(random_state=42),
                    {
                        'alpha': [0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
                    }
                )
            }

    def _generate_optimization_insights(self, results, problem_type):
        """Generate insights from optimization results"""
        insights = []

        # Count successful optimizations
        successful = [k for k, v in results.items() if 'error' not in v]
        failed = [k for k, v in results.items() if 'error' in v]

        insights.append(f"Successfully optimized {len(successful)} out of {len(results)} models")

        if failed:
            insights.append(f"Failed models: {', '.join(failed)}")

        # Performance insights
        if successful:
            if 'classification' in problem_type:
                scores = [results[model]['test_score'] for model in successful]
                best_score = max(scores)
                worst_score = min(scores)
                insights.append(f"Test accuracy range: {worst_score:.3f} - {best_score:.3f}")

                if best_score > 0.9:
                    insights.append("Excellent performance achieved through optimization")
                elif best_score > 0.8:
                    insights.append("Good performance achieved through optimization")
            else:
                scores = [results[model]['test_score'] for model in successful]
                best_score = min(scores)  # Lower is better for RMSE
                worst_score = max(scores)
                insights.append(f"Test RMSE range: {best_score:.3f} - {worst_score:.3f}")

        # Parameter insights
        param_insights = []
        for model_name, result in results.items():
            if 'best_params' in result:
                best_params = result['best_params']
                for param, value in best_params.items():
                    param_insights.append(f"{model_name}: {param} = {value}")

        if param_insights:
            insights.append("Key optimized parameters:")
            insights.extend(param_insights[:5])  # Show top 5

        return insights

    def feature_selection_optimization(self, data, target_column, n_features_range=(5, 20)):
        """Optimize feature selection along with model parameters"""
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        from sklearn.pipeline import Pipeline

        X = data.drop(columns=[target_column])
        y = data[target_column]

        problem_type = self._detect_problem_type(y)
        X_processed = self._preprocess_features(X)

        # Create pipeline with feature selection
        if 'classification' in problem_type:
            selector = SelectKBest(score_func=f_classif)
            base_model = RandomForestClassifier(random_state=42)
            scoring = 'accuracy'
        else:
            selector = SelectKBest(score_func=f_regression)
            base_model = RandomForestRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', selector),
            ('model', base_model)
        ])

        # Parameter grid including feature selection
        param_grid = {
            'selector__k': list(range(n_features_range[0], min(n_features_range[1], X_processed.shape[1]))),
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20]
        }

        # Optimize
        search = GridSearchCV(
            pipeline, param_grid,
            cv=5, scoring=scoring,
            n_jobs=-1, verbose=0
        )

        search.fit(X_processed, y)

        # Get selected features
        best_selector = search.best_estimator_['selector']
        selected_features = X_processed.columns[best_selector.get_support()].tolist()

        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'selected_features': selected_features,
            'n_selected_features': len(selected_features)
        }

    def multi_objective_optimization(self, data, target_column, objectives=['accuracy', 'speed']):
        """Multi-objective optimization considering performance and speed"""
        import time

        X = data.drop(columns=[target_column])
        y = data[target_column]

        problem_type = self._detect_problem_type(y)
        X_processed = self._preprocess_features(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models with different speed/accuracy trade-offs
        models = {
            'Fast - Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Medium - Random Forest (Small)': RandomForestClassifier(n_estimators=50, random_state=42),
            'Slow - Random Forest (Large)': RandomForestClassifier(n_estimators=200, random_state=42)
        }

        results = {}

        for name, model in models.items():
            start_time = time.time()

            # Train
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # Predict
            start_time = time.time()
            y_pred = model.predict(X_test_scaled)
            predict_time = time.time() - start_time

            # Calculate metrics
            if 'classification' in problem_type:
                performance = accuracy_score(y_test, y_pred)
            else:
                performance = -mean_squared_error(y_test, y_pred, squared=False)  # Negative for maximization

            results[name] = {
                'performance': performance,
                'train_time': train_time,
                'predict_time': predict_time,
                'total_time': train_time + predict_time,
                'model': model
            }

        # Calculate Pareto frontier
        pareto_optimal = self._find_pareto_optimal(results, objectives)

        return {
            'all_results': results,
            'pareto_optimal': pareto_optimal,
            'objectives': objectives,
            'recommendation': self._recommend_based_on_objectives(pareto_optimal, objectives)
        }

    def _find_pareto_optimal(self, results, objectives):
        """Find Pareto optimal solutions for multi-objective optimization"""
        pareto_optimal = []

        for name1, result1 in results.items():
            is_dominated = False

            for name2, result2 in results.items():
                if name1 != name2:
                    # Check if result2 dominates result1
                    dominates = True
                    for obj in objectives:
                        if obj == 'accuracy':
                            if result2['performance'] <= result1['performance']:
                                dominates = False
                                break
                        elif obj == 'speed':
                            if result2['total_time'] >= result1['total_time']:
                                dominates = False
                                break

                    if dominates:
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_optimal.append(name1)

        return pareto_optimal

    def _recommend_based_on_objectives(self, pareto_optimal, objectives):
        """Recommend best model based on objectives"""
        if len(pareto_optimal) == 1:
            return {
                'model': pareto_optimal[0],
                'reason': 'Single Pareto optimal solution'
            }
        elif 'accuracy' in objectives and 'speed' in objectives:
            return {
                'model': pareto_optimal[0],  # First in Pareto set
                'reason': 'Best balance between accuracy and speed',
                'alternatives': pareto_optimal[1:] if len(pareto_optimal) > 1 else []
            }
        else:
            return {
                'model': pareto_optimal[0],
                'reason': 'Top Pareto optimal solution',
                'alternatives': pareto_optimal[1:] if len(pareto_optimal) > 1 else []
            }

    def generate_automl_report(self, optimization_results):
        """Generate comprehensive AutoML report"""
        if optimization_results['status'] != 'success':
            return f"AutoML failed: {optimization_results.get('error', 'Unknown error')}"

        report = []

        # Header
        report.append("=" * 50)
        report.append("AUTOMATED MACHINE LEARNING REPORT")
        report.append("=" * 50)

        # Problem summary
        best_model = optimization_results['best_model']
        report.append(f"\nProblem Type: {optimization_results['problem_type']}")
        report.append(f"Optimization Metric: {optimization_results['optimization_metric']}")
        report.append(f"Best Model: {best_model['name']}")
        report.append(f"Best Score: {best_model['score']:.4f}")

        # Model parameters
        report.append(f"\nOptimized Parameters:")
        for param, value in best_model['best_params'].items():
            report.append(f"  - {param}: {value}")

        # All models performance
        report.append(f"\nAll Models Performance:")
        for model_name, result in optimization_results['all_results'].items():
            if 'error' not in result:
                report.append(f"  - {model_name}: {result['test_score']:.4f}")
            else:
                report.append(f"  - {model_name}: FAILED ({result['error']})")

        # Insights
        report.append(f"\nKey Insights:")
        for insight in optimization_results['insights']:
            report.append(f"  • {insight}")

        # Preprocessing info
        preprocessing = optimization_results['preprocessing_info']
        report.append(f"\nPreprocessing:")
        report.append(f"  - Original features: {preprocessing['original_features']}")
        report.append(f"  - Processed features: {preprocessing['features_processed']}")
        report.append(f"  - Scaler: {preprocessing['scaler_used']}")

        return "\n".join(report)
