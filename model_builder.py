"""
Model Building Agent - Handles comprehensive model selection and building
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA

# Classification algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Regression algorithms
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           silhouette_score, adjusted_rand_score, roc_curve, precision_recall_curve)

# Optional imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding, Flatten
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ModelBuildingAgent:
    """Agent responsible for comprehensive model selection and building"""

    def __init__(self):
        self.models = {}
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.label_encoders = {}
        self.feature_selector = None
        self.preprocessing_pipeline = {}

    def build_model(self, data, target_column, problem_type=None, model_categories=['traditional_ml']):
        """
        Build and evaluate comprehensive set of ML models
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            problem_type: 'classification', 'regression', or None (auto-detect)
            model_categories: List of model types to train
        Returns:
            Dictionary with model results and recommendations
        """
        if target_column not in data.columns:
            return {'status': 'error', 'error': f'Target column {target_column} not found'}

        print(f"🤖 Building models for {target_column}...")

        try:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect problem type if not specified
            if problem_type is None:
                problem_type = self._detect_problem_type(y)

            print(f"📊 Detected problem type: {problem_type}")

            # Preprocess features
            X_processed = self._preprocess_features(X)

            # Encode target if classification
            if 'classification' in problem_type:
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    self.label_encoders['target'] = le
                else:
                    y_encoded = y.copy()
            else:
                y_encoded = y.copy()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42,
                stratify=y_encoded if 'classification' in problem_type else None
            )

            # Feature scaling
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)

            # Build models based on categories
            all_results = {}

            if 'traditional_ml' in model_categories:
                print("🔄 Training traditional ML models...")
                ml_results = self._build_traditional_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
                all_results.update(ml_results)

            if 'ensemble' in model_categories:
                print("🔄 Training ensemble models...")
                ensemble_results = self._build_ensemble_models(X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
                all_results.update(ensemble_results)

            if 'boosting' in model_categories:
                print("🔄 Training boosting models...")
                boosting_results = self._build_boosting_models(X_train, X_test, y_train, y_test, problem_type)
                all_results.update(boosting_results)

            if 'deep_learning' in model_categories and TENSORFLOW_AVAILABLE:
                print("🔄 Training deep learning models...")
                dl_results = self._build_deep_learning_models(X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
                all_results.update(dl_results)

            if 'clustering' in model_categories and problem_type == 'unsupervised':
                print("🔄 Training clustering models...")
                cluster_results = self._build_clustering_models(X_train_scaled)
                all_results.update(cluster_results)

            # Filter successful models
            valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
            if not valid_results:
                return {'status': 'error', 'error': 'No models trained successfully'}

            # Select best model
            best_model_name = self._select_best_model(valid_results, problem_type)

            # Generate model insights
            model_insights = self._generate_model_insights(valid_results, problem_type)

            return {
                'status': 'success',
                'problem_type': problem_type,
                'results': all_results,
                'best_model': best_model_name,
                'best_model_details': valid_results.get(best_model_name, {}),
                'feature_importance': self._get_feature_importance(valid_results.get(best_model_name, {}).get('model'), X.columns),
                'model_comparison': self._create_model_comparison(valid_results, problem_type),
                'model_insights': model_insights,
                'preprocessing_info': {
                    'scaler_used': 'StandardScaler',
                    'features_processed': X_processed.shape[1],
                    'original_features': X.shape[1],
                    'target_encoded': 'target' in self.label_encoders
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'details': 'Error occurred during model building process'
            }

    def _detect_problem_type(self, target):
        """Detect problem type with enhanced logic"""
        unique_count = target.nunique()

        if target.dtype == 'object':
            return 'classification'
        elif unique_count == 2:
            return 'binary_classification'
        elif unique_count < 20 and target.dtype in ['int64', 'int32']:
            # Check if it's actually categorical
            if sorted(target.unique()) == list(range(unique_count)):
                return 'multiclass_classification'
            else:
                return 'regression'
        else:
            return 'regression'

    def _preprocess_features(self, X):
        """Advanced feature preprocessing with detailed tracking"""
        X_processed = X.copy()
        preprocessing_steps = []

        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            unique_count = X_processed[col].nunique()

            if unique_count <= 10:
                # One-hot encode low cardinality
                dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                X_processed = pd.concat([X_processed, dummies], axis=1)
                X_processed.drop(columns=[col], inplace=True)
                preprocessing_steps.append(f'One-hot encoded {col}')
            else:
                # Label encode high cardinality
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
                preprocessing_steps.append(f'Label encoded {col}')

        # Handle missing values in numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_processed[col].isnull().any():
                X_processed[col].fillna(X_processed[col].median(), inplace=True)
                preprocessing_steps.append(f'Filled missing values in {col}')

        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())

        self.preprocessing_pipeline['steps'] = preprocessing_steps

        return X_processed

    def _scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)

        return X_train_scaled, X_test_scaled

    def _build_traditional_ml_models(self, X_train, X_test, y_train, y_test, problem_type):
        """Build traditional machine learning models"""
        results = {}

        if 'classification' in problem_type:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
                'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                'Ridge Classifier': RidgeClassifier(random_state=42),
                'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000)
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'Elastic Net': ElasticNet(random_state=42),
                'SVR (RBF)': SVR(kernel='rbf'),
                'SVR (Linear)': SVR(kernel='linear'),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Bayesian Ridge': BayesianRidge(),
                'Huber Regressor': HuberRegressor()
            }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if 'classification' in problem_type:
                    metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'model_type': 'traditional_ml',
                    **metrics
                }
            except Exception as e:
                results[name] = {'error': str(e), 'model_type': 'traditional_ml'}

        return results

    def _build_ensemble_models(self, X_train, X_test, y_train, y_test, problem_type):
        """Build ensemble models"""
        results = {}

        if 'classification' in problem_type:
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if 'classification' in problem_type:
                    metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'model_type': 'ensemble',
                    **metrics
                }
            except Exception as e:
                results[name] = {'error': str(e), 'model_type': 'ensemble'}

        return results

    def _build_boosting_models(self, X_train, X_test, y_train, y_test, problem_type):
        """Build advanced boosting models"""
        results = {}

        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                if 'classification' in problem_type:
                    if problem_type == 'binary_classification':
                        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                    else:
                        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
                else:
                    xgb_model = xgb.XGBRegressor(random_state=42)

                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)

                if 'classification' in problem_type:
                    metrics = self._calculate_classification_metrics(y_test, y_pred, xgb_model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                results['XGBoost'] = {
                    'model': xgb_model,
                    'predictions': y_pred,
                    'model_type': 'boosting',
                    **metrics
                }
            except Exception as e:
                results['XGBoost'] = {'error': str(e), 'model_type': 'boosting'}

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                if 'classification' in problem_type:
                    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                else:
                    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)

                lgb_model.fit(X_train, y_train)
                y_pred = lgb_model.predict(X_test)

                if 'classification' in problem_type:
                    metrics = self._calculate_classification_metrics(y_test, y_pred, lgb_model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                results['LightGBM'] = {
                    'model': lgb_model,
                    'predictions': y_pred,
                    'model_type': 'boosting',
                    **metrics
                }
            except Exception as e:
                results['LightGBM'] = {'error': str(e), 'model_type': 'boosting'}

        # CatBoost
        if CATBOOST_AVAILABLE:
            try:
                if 'classification' in problem_type:
                    cat_model = cb.CatBoostClassifier(random_state=42, verbose=False)
                else:
                    cat_model = cb.CatBoostRegressor(random_state=42, verbose=False)

                cat_model.fit(X_train, y_train)
                y_pred = cat_model.predict(X_test)

                if 'classification' in problem_type:
                    metrics = self._calculate_classification_metrics(y_test, y_pred, cat_model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                results['CatBoost'] = {
                    'model': cat_model,
                    'predictions': y_pred,
                    'model_type': 'boosting',
                    **metrics
                }
            except Exception as e:
                results['CatBoost'] = {'error': str(e), 'model_type': 'boosting'}

        return results

    def _build_deep_learning_models(self, X_train, X_test, y_train, y_test, problem_type):
        """Build deep learning models using TensorFlow/Keras"""
        results = {}

        if not TENSORFLOW_AVAILABLE:
            return results

        input_dim = X_train.shape[1]

        # Simple MLP
        try:
            model = self._create_simple_mlp(input_dim, problem_type, len(np.unique(y_train)) if 'classification' in problem_type else 1)

            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]

            # Prepare target for deep learning
            if 'classification' in problem_type:
                n_classes = len(np.unique(y_train))
                if n_classes > 2:
                    y_train_dl = to_categorical(y_train)
                    y_test_dl = to_categorical(y_test)
                else:
                    y_train_dl = y_train
                    y_test_dl = y_test
            else:
                y_train_dl = y_train
                y_test_dl = y_test

            # Train model
            history = model.fit(
                X_train, y_train_dl,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            # Predictions
            if 'classification' in problem_type:
                y_pred_proba = model.predict(X_test)
                if len(np.unique(y_train)) > 2:
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

                metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test, y_pred_proba)
            else:
                y_pred = model.predict(X_test).flatten()
                metrics = self._calculate_regression_metrics(y_test, y_pred)

            results['Deep Learning - MLP'] = {
                'model': model,
                'predictions': y_pred,
                'model_type': 'deep_learning',
                'training_history': history.history,
                **metrics
            }

        except Exception as e:
            results['Deep Learning - MLP'] = {'error': str(e), 'model_type': 'deep_learning'}

        return results

    def _build_clustering_models(self, X_train):
        """Build clustering models for unsupervised learning"""
        results = {}

        models = {
            'K-Means': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=3),
            'Gaussian Mixture': GaussianMixture(n_components=3, random_state=42)
        }

        for name, model in models.items():
            try:
                cluster_labels = model.fit_predict(X_train)

                # Calculate clustering metrics
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(X_train, cluster_labels)

                    results[name] = {
                        'model': model,
                        'cluster_labels': cluster_labels,
                        'silhouette_score': silhouette,
                        'n_clusters': len(np.unique(cluster_labels)),
                        'model_type': 'clustering'
                    }
                else:
                    results[name] = {'error': 'All points assigned to single cluster', 'model_type': 'clustering'}

            except Exception as e:
                results[name] = {'error': str(e), 'model_type': 'clustering'}

        return results

    def _create_simple_mlp(self, input_dim, problem_type, output_dim):
        """Create simple Multi-Layer Perceptron"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu')
        ])

        if 'classification' in problem_type:
            if output_dim == 2:
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.add(Dense(output_dim, activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    def _calculate_classification_metrics(self, y_true, y_pred, model=None, X_test=None, y_pred_proba=None):
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except:
            pass

        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                if y_pred_proba is not None:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                elif hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass

        # Classification report
        try:
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except:
            pass

        return metrics

    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate comprehensive regression metrics"""
        metrics = {
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

        # Additional regression metrics
        try:
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
        except:
            pass

        return metrics

    def _select_best_model(self, results, problem_type):
        """Select the best model based on problem type"""
        if 'classification' in problem_type:
            # Prioritize models with highest accuracy
            valid_models = {k: v for k, v in results.items() if 'accuracy' in v}
            if valid_models:
                return max(valid_models.keys(), key=lambda x: valid_models[x]['accuracy'])
        else:
            # Prioritize models with lowest RMSE
            valid_models = {k: v for k, v in results.items() if 'rmse' in v}
            if valid_models:
                return min(valid_models.keys(), key=lambda x: valid_models[x]['rmse'])

        # Fallback to first successful model
        return list(results.keys())[0] if results else None

    def _create_model_comparison(self, results, problem_type):
        """Create model comparison summary"""
        comparison = {}

        for model_name, result in results.items():
            if 'error' not in result:
                if 'classification' in problem_type:
                    comparison[model_name] = {
                        'accuracy': result.get('accuracy', 0),
                        'f1_score': result.get('f1_score', 0),
                        'precision': result.get('precision', 0),
                        'recall': result.get('recall', 0),
                        'model_type': result.get('model_type', 'unknown')
                    }
                    if 'roc_auc' in result:
                        comparison[model_name]['roc_auc'] = result['roc_auc']
                else:
                    comparison[model_name] = {
                        'rmse': result.get('rmse', float('inf')),
                        'mae': result.get('mae', float('inf')),
                        'r2_score': result.get('r2_score', 0),
                        'model_type': result.get('model_type', 'unknown')
                    }
                    if 'mape' in result:
                        comparison[model_name]['mape'] = result['mape']

        return comparison

    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from various model types"""
        if model is None:
            return {}

        try:
            # Tree-based models
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            # Linear models
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    # Multi-class classification
                    importance = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
                else:
                    importance = dict(zip(feature_names, np.abs(model.coef_)))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            # XGBoost
            elif hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return {}

    def _generate_model_insights(self, results, problem_type):
        """Generate insights about model performance"""
        insights = []

        # Performance insights
        if 'classification' in problem_type:
            accuracies = [r['accuracy'] for r in results.values() if 'accuracy' in r]
            if accuracies:
                best_acc = max(accuracies)
                worst_acc = min(accuracies)
                insights.append(f"Accuracy range: {worst_acc:.3f} - {best_acc:.3f}")

                if best_acc > 0.9:
                    insights.append("Excellent model performance achieved")
                elif best_acc > 0.8:
                    insights.append("Good model performance achieved")
                else:
                    insights.append("Model performance could be improved")
        else:
            r2_scores = [r['r2_score'] for r in results.values() if 'r2_score' in r]
            if r2_scores:
                best_r2 = max(r2_scores)
                insights.append(f"Best R² score: {best_r2:.3f}")

                if best_r2 > 0.8:
                    insights.append("Strong predictive power achieved")
                elif best_r2 > 0.6:
                    insights.append("Moderate predictive power achieved")
                else:
                    insights.append("Weak predictive power - consider feature engineering")

        # Model type insights
        model_types = {}
        for result in results.values():
            if 'model_type' in result:
                model_type = result['model_type']
                model_types[model_type] = model_types.get(model_type, 0) + 1

        if 'ensemble' in model_types or 'boosting' in model_types:
            insights.append("Tree-based models are performing well on this dataset")

        if 'deep_learning' in model_types:
            insights.append("Deep learning models were successfully trained")

        return insights
