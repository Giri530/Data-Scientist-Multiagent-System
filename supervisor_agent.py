"""
Supervisor Agent - Main orchestrator for the entire data science pipeline
"""

import pandas as pd
import numpy as np
from data_loader import DataLoaderAgent
from data_cleaner import DataCleaningAgent
from eda_agent import EDAAgent
from domain_expert import DomainExpertAgent
from model_builder import ModelBuildingAgent
from automl_agent import AutoMLAgent


class SupervisorAgent:
    """Main supervisor agent that orchestrates the entire pipeline"""

    def __init__(self):
        self.data_loader = DataLoaderAgent()
        self.data_cleaner = DataCleaningAgent()
        self.eda_agent = EDAAgent()
        self.domain_expert = DomainExpertAgent()
        self.model_builder = ModelBuildingAgent()
        self.automl_agent = AutoMLAgent()

        self.pipeline_state = {
            'current_step': 'initialized',
            'completed_steps': [],
            'results': {},
            'errors': []
        }

        self.pipeline_config = {
            'data_cleaning': {
                'aggressive_cleaning': False,
                'handle_outliers': True
            },
            'modeling': {
                'categories': ['traditional_ml', 'ensemble', 'boosting'],
                'enable_automl': True,
                'automl_time_budget': 300
            },
            'output': {
                'generate_visualizations': True,
                'create_report': True
            }
        }

    def execute_pipeline(self, data_source, source_type='csv', target_column=None,
                        domain=None, pipeline_config=None, **kwargs):
        """
        Execute the complete end-to-end data science pipeline
        Args:
            data_source: Path to data file or data source
            source_type: Type of data source ('csv', 'json', etc.)
            target_column: Name of target variable for supervised learning
            domain: Domain hint ('finance', 'healthcare', etc.)
            pipeline_config: Configuration dictionary for pipeline steps
            **kwargs: Additional parameters for data loading
        Returns:
            Comprehensive pipeline results
        """
        try:
            print("🚀 Starting End-to-End Data Science Pipeline...")
            print("=" * 60)

            # Update configuration if provided
            if pipeline_config:
                self.pipeline_config.update(pipeline_config)

            # Step 1: Data Loading
            print("📁 Step 1: Loading data...")
            load_result = self._execute_data_loading(data_source, source_type, **kwargs)
            if load_result['status'] != 'success':
                return self._handle_pipeline_error('data_loading', load_result)

            data = load_result['data']
            print(f"✅ Data loaded successfully. Shape: {data.shape}")
            print(f"   Columns: {', '.join(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}")

            # Step 2: Data Cleaning
            print("\n🧹 Step 2: Cleaning data...")
            clean_result = self._execute_data_cleaning(data)
            if clean_result['status'] != 'success':
                return self._handle_pipeline_error('data_cleaning', clean_result)

            cleaned_data = clean_result['data']
            cleaning_report = clean_result['cleaning_report']
            print(f"✅ Data cleaned successfully. New shape: {cleaned_data.shape}")
            print(f"   Removed {cleaning_report.get('duplicates_removed', 0)} duplicates")
            print(f"   Handled {len(cleaning_report.get('missing_values', {}))} columns with missing values")

            # Step 3: Exploratory Data Analysis
            print("\n📊 Step 3: Performing EDA...")
            eda_result = self._execute_eda(cleaned_data, target_column)
            print("✅ EDA completed successfully")
            eda_insights = eda_result.get('analysis', {}).get('feature_insights', [])
            if eda_insights:
                print(f"   Found {len(eda_insights)} key insights")

            # Step 4: Domain Expert Analysis
            print("\n🎓 Step 4: Getting domain insights...")
            domain_result = self._execute_domain_analysis(cleaned_data, domain, target_column)
            detected_domain = domain_result['detected_domain']
            confidence = domain_result['confidence']
            print(f"✅ Domain analysis completed")
            print(f"   Detected domain: {detected_domain} (confidence: {confidence:.2f})")
            print(f"   Generated {len(domain_result['recommendations'])} recommendations")

            # Step 5: Model Building (if target specified)
            model_result = None
            automl_result = None

            if target_column and target_column in cleaned_data.columns:
                print(f"\n🤖 Step 5: Building models for target '{target_column}'...")

                # Traditional model building
                model_result = self._execute_model_building(cleaned_data, target_column)

                if model_result['status'] == 'success':
                    best_model = model_result['best_model']
                    problem_type = model_result['problem_type']
                    print(f"✅ Models built successfully")
                    print(f"   Problem type: {problem_type}")
                    print(f"   Best model: {best_model}")

                    # AutoML optimization if enabled
                    if self.pipeline_config['modeling']['enable_automl']:
                        print(f"\n🔧 Step 5b: AutoML optimization...")
                        automl_result = self._execute_automl(cleaned_data, target_column)

                        if automl_result['status'] == 'success':
                            automl_best = automl_result['best_model']['name']
                            automl_score = automl_result['best_model']['score']
                            print(f"✅ AutoML optimization completed")
                            print(f"   Best optimized model: {automl_best} (score: {automl_score:.4f})")
                        else:
                            print(f"⚠️ AutoML optimization failed: {automl_result.get('error', 'Unknown error')}")
                else:
                    print(f"⚠️ Model building failed: {model_result.get('error', 'Unknown error')}")
            else:
                if target_column:
                    print(f"\n⚠️ Target column '{target_column}' not found in data")
                else:
                    print(f"\n💡 No target column specified - skipping supervised learning")

            # Step 6: Generate Final Report
            print(f"\n📈 Step 6: Generating comprehensive report...")
            final_report = self._generate_final_report(
                load_result, clean_result, eda_result, domain_result,
                model_result, automl_result, cleaned_data, target_column
            )
            print("✅ Report generated successfully")

            print("\n🎉 Pipeline completed successfully!")
            print("=" * 60)

            return {
                'status': 'success',
                'pipeline_results': self.pipeline_state['results'],
                'final_report': final_report,
                'data_shape': cleaned_data.shape,
                'target_column': target_column,
                'best_model': model_result['best_model'] if model_result and model_result['status'] == 'success' else None,
                'automl_best': automl_result['best_model'] if automl_result and automl_result['status'] == 'success' else None
            }

        except Exception as e:
            error_info = {
                'status': 'error',
                'error': str(e),
                'step': self.pipeline_state['current_step'],
                'completed_steps': self.pipeline_state['completed_steps']
            }
            print(f"\n❌ Pipeline failed at step: {self.pipeline_state['current_step']}")
            print(f"   Error: {str(e)}")
            return error_info

    def _execute_data_loading(self, data_source, source_type, **kwargs):
        """Execute data loading step"""
        self.pipeline_state['current_step'] = 'data_loading'

        result = self.data_loader.load_data(data_source, source_type, **kwargs)
        self.pipeline_state['results']['data_loading'] = result

        if result['status'] == 'success':
            self.pipeline_state['completed_steps'].append('data_loading')

        return result

    def _execute_data_cleaning(self, data):
        """Execute data cleaning step"""
        self.pipeline_state['current_step'] = 'data_cleaning'

        cleaning_config = self.pipeline_config['data_cleaning']
        result = self.data_cleaner.clean_data(
            data,
            aggressive_cleaning=cleaning_config['aggressive_cleaning']
        )
        self.pipeline_state['results']['data_cleaning'] = result

        if result['status'] == 'success':
            self.pipeline_state['completed_steps'].append('data_cleaning')

        return result

    def _execute_eda(self, data, target_column=None):
        """Execute EDA step"""
        self.pipeline_state['current_step'] = 'eda'

        result = self.eda_agent.analyze_data(data, target_column)
        self.pipeline_state['results']['eda'] = result

        if result['status'] == 'success':
            self.pipeline_state['completed_steps'].append('eda')

        return result

    def _execute_domain_analysis(self, data, domain=None, target_column=None):
        """Execute domain expert analysis step"""
        self.pipeline_state['current_step'] = 'domain_analysis'

        result = self.domain_expert.provide_domain_insights(data, domain, target_column)
        self.pipeline_state['results']['domain_analysis'] = result

        self.pipeline_state['completed_steps'].append('domain_analysis')
        return result

    def _execute_model_building(self, data, target_column):
        """Execute model building step"""
        self.pipeline_state['current_step'] = 'model_building'

        modeling_config = self.pipeline_config['modeling']
        result = self.model_builder.build_model(
            data,
            target_column,
            model_categories=modeling_config['categories']
        )
        self.pipeline_state['results']['model_building'] = result

        if result['status'] == 'success':
            self.pipeline_state['completed_steps'].append('model_building')

        return result

    def _execute_automl(self, data, target_column):
        """Execute AutoML optimization step"""
        self.pipeline_state['current_step'] = 'automl'

        modeling_config = self.pipeline_config['modeling']
        result = self.automl_agent.auto_optimize(
            data,
            target_column,
            time_budget=modeling_config['automl_time_budget']
        )
        self.pipeline_state['results']['automl'] = result

        if result['status'] == 'success':
            self.pipeline_state['completed_steps'].append('automl')

        return result

    def _handle_pipeline_error(self, step, error_result):
        """Handle pipeline errors gracefully"""
        self.pipeline_state['errors'].append({
            'step': step,
            'error': error_result.get('error', 'Unknown error')
        })

        return {
            'status': 'error',
            'failed_step': step,
            'error': error_result.get('error', 'Unknown error'),
            'completed_steps': self.pipeline_state['completed_steps'],
            'partial_results': self.pipeline_state['results']
        }

    def _generate_final_report(self, load_result, clean_result, eda_result,
                              domain_result, model_result, automl_result,
                              data, target_column):
        """Generate comprehensive final report"""

        report = {
            'executive_summary': self._generate_executive_summary(
                data, target_column, model_result, automl_result
            ),
            'data_overview': self._generate_data_overview(load_result, clean_result, data),
            'exploratory_analysis': self._generate_eda_summary(eda_result),
            'domain_insights': self._generate_domain_summary(domain_result),
            'modeling_results': self._generate_modeling_summary(model_result, automl_result),
            'recommendations': self._generate_recommendations(
                domain_result, model_result, automl_result
            ),
            'technical_details': {
                'pipeline_config': self.pipeline_config,
                'completed_steps': self.pipeline_state['completed_steps'],
                'processing_time': 'Not tracked',  # Could add timing
                'data_quality_score': self._calculate_data_quality_score(data)
            }
        }

        return report

    def _generate_executive_summary(self, data, target_column, model_result, automl_result):
        """Generate executive summary"""
        summary = []

        # Data summary
        summary.append(f"Analyzed dataset with {data.shape[0]:,} rows and {data.shape[1]} features")

        # Problem type and target
        if target_column and model_result and model_result['status'] == 'success':
            problem_type = model_result['problem_type']
            best_model = model_result['best_model']

            if 'classification' in problem_type:
                best_score = model_result['results'][best_model]['accuracy']
                summary.append(f"Built {problem_type} models with best accuracy of {best_score:.3f}")
            else:
                best_score = model_result['results'][best_model]['r2_score']
                summary.append(f"Built {problem_type} models with best R² score of {best_score:.3f}")

            summary.append(f"Best performing model: {best_model}")

            # AutoML results
            if automl_result and automl_result['status'] == 'success':
                automl_model = automl_result['best_model']['name']
                automl_score = automl_result['best_model']['score']
                summary.append(f"AutoML optimization improved performance to {automl_score:.3f} using {automl_model}")

        return summary

    def _generate_data_overview(self, load_result, clean_result, data):
        """Generate data overview section"""
        overview = {}

        if load_result['status'] == 'success':
            original_info = load_result['info']
            overview['original_shape'] = original_info['shape']
            overview['memory_usage'] = original_info.get('memory_usage', 'Unknown')

        if clean_result['status'] == 'success':
            cleaning_report = clean_result['cleaning_report']
            overview['final_shape'] = data.shape
            overview['cleaning_summary'] = {
                'duplicates_removed': cleaning_report.get('duplicates_removed', 0),
                'missing_values_handled': len(cleaning_report.get('missing_values', {})),
                'outliers_handled': len(cleaning_report.get('outliers', {}))
            }

        # Data types
        overview['data_types'] = {
            'numeric': len(data.select_dtypes(include=[np.number]).columns),
            'categorical': len(data.select_dtypes(include=['object']).columns),
            'datetime': len(data.select_dtypes(include=['datetime64']).columns)
        }

        return overview

    def _generate_eda_summary(self, eda_result):
        """Generate EDA summary"""
        if eda_result['status'] != 'success':
            return {'error': 'EDA analysis failed'}

        analysis = eda_result['analysis']
        summary = {}

        # Key insights
        if 'feature_insights' in analysis:
            insights = analysis['feature_insights']
            summary['key_insights'] = [insight['insight'] for insight in insights[:5]]

        # Correlations
        if 'correlations' in analysis:
            corr_info = analysis['correlations']
            if 'strong_correlations' in corr_info:
                strong_corr = corr_info['strong_correlations']
                summary['strong_correlations'] = len(strong_corr)
                if strong_corr:
                    summary['top_correlations'] = [
                        f"{item['var1']} - {item['var2']}: {item['correlation']:.3f}"
                        for item in strong_corr[:3]
                    ]

        return summary

    def _generate_domain_summary(self, domain_result):
        """Generate domain analysis summary"""
        summary = {
            'detected_domain': domain_result['detected_domain'],
            'confidence': domain_result['confidence'],
            'key_insights': domain_result['insights'][:3],
            'recommendations': domain_result['recommendations'][:5],
            'feature_engineering_suggestions': domain_result['feature_engineering_suggestions'][:3]
        }

        return summary

    def _generate_modeling_summary(self, model_result, automl_result):
        """Generate modeling results summary"""
        summary = {}

        if model_result and model_result['status'] == 'success':
            summary['traditional_ml'] = {
                'problem_type': model_result['problem_type'],
                'best_model': model_result['best_model'],
                'models_trained': len([k for k, v in model_result['results'].items() if 'error' not in v]),
                'model_comparison': model_result['model_comparison']
            }

            # Feature importance
            if model_result['feature_importance']:
                top_features = list(model_result['feature_importance'].items())[:5]
                summary['traditional_ml']['top_features'] = [
                    f"{feature}: {importance:.3f}" for feature, importance in top_features
                ]

        if automl_result and automl_result['status'] == 'success':
            best_model = automl_result['best_model']
            summary['automl'] = {
                'best_model': best_model['name'],
                'best_score': best_model['score'],
                'optimization_metric': automl_result['optimization_metric'],
                'models_optimized': len([k for k, v in automl_result['all_results'].items() if 'error' not in v]),
                'best_parameters': best_model['best_params']
            }

        return summary

    def _generate_recommendations(self, domain_result, model_result, automl_result):
        """Generate final recommendations"""
        recommendations = []

        # Domain-specific recommendations
        domain_recs = domain_result['recommendations'][:3]
        recommendations.extend([f"Domain: {rec}" for rec in domain_recs])

        # Modeling recommendations
        if model_result and model_result['status'] == 'success':
            modeling_recs = domain_result['modeling_recommendations'][:2]
            recommendations.extend([f"Modeling: {rec}" for rec in modeling_recs])

        # Feature engineering recommendations
        fe_recs = domain_result['feature_engineering_suggestions'][:2]
        recommendations.extend([f"Feature Engineering: {rec}" for rec in fe_recs])

        # Performance recommendations
        if automl_result and automl_result['status'] == 'success':
            automl_insights = automl_result['insights'][:2]
            recommendations.extend([f"AutoML: {insight}" for insight in automl_insights])

        return recommendations

    def _calculate_data_quality_score(self, data):
        """Calculate overall data quality score"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()

        # Basic quality score based on completeness
        completeness_score = (total_cells - missing_cells) / total_cells

        # Adjust for duplicates
        duplicate_penalty = data.duplicated().sum() / len(data)

        # Adjust for constant columns
        constant_penalty = sum(data.nunique() == 1) / len(data.columns)

        quality_score = completeness_score * (1 - duplicate_penalty) * (1 - constant_penalty)

        return min(max(quality_score, 0), 1)  # Clamp between 0 and 1

    def generate_pipeline_summary(self, pipeline_results):
        """Generate a concise pipeline summary"""
        if pipeline_results['status'] != 'success':
            return f"Pipeline failed: {pipeline_results.get('error', 'Unknown error')}"

        summary_lines = []

        # Header
        summary_lines.append("🔍 DATA SCIENCE PIPELINE SUMMARY")
        summary_lines.append("=" * 40)

        # Data info
        data_shape = pipeline_results['data_shape']
        summary_lines.append(f"📊 Dataset: {data_shape[0]:,} rows × {data_shape[1]} columns")

        # Target and problem type
        target = pipeline_results.get('target_column')
        if target:
            summary_lines.append(f"🎯 Target: {target}")

            # Model performance
            best_model = pipeline_results.get('best_model')
            if best_model:
                summary_lines.append(f"🤖 Best Model: {best_model}")

            # AutoML results
            automl_best = pipeline_results.get('automl_best')
            if automl_best:
                automl_name = automl_best['name']
                automl_score = automl_best['score']
                summary_lines.append(f"🔧 AutoML Best: {automl_name} ({automl_score:.4f})")
        else:
            summary_lines.append("💡 Exploratory analysis completed (no target specified)")

        # Key insights
        final_report = pipeline_results.get('final_report', {})
        exec_summary = final_report.get('executive_summary', [])
        if exec_summary:
            summary_lines.append("\n📋 Key Findings:")
            for insight in exec_summary[:3]:
                summary_lines.append(f"  • {insight}")

        # Recommendations
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            summary_lines.append(f"\n💡 Top Recommendations:")
            for rec in recommendations[:3]:
                summary_lines.append(f"  • {rec}")

        return "\n".join(summary_lines)

    def export_results(self, pipeline_results, export_format='json', file_path=None):
        """Export pipeline results to various formats"""
        if pipeline_results['status'] != 'success':
            raise ValueError("Cannot export failed pipeline results")

        export_data = {
            'pipeline_summary': {
                'status': pipeline_results['status'],
                'data_shape': pipeline_results['data_shape'],
                'target_column': pipeline_results['target_column'],
                'completion_time': 'Not tracked'  # Could add timestamp
            },
            'final_report': pipeline_results['final_report'],
            'model_results': pipeline_results['pipeline_results'].get('model_building', {}),
            'automl_results': pipeline_results['pipeline_results'].get('automl', {})
        }

        if export_format.lower() == 'json':
            import json
            output = json.dumps(export_data, indent=2, default=str)
        elif export_format.lower() == 'yaml':
            try:
                import yaml
                output = yaml.dump(export_data, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML export")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
            return f"Results exported to {file_path}"
        else:
            return output

    def get_pipeline_status(self):
        """Get current pipeline status"""
        return {
            'current_step': self.pipeline_state['current_step'],
            'completed_steps': self.pipeline_state['completed_steps'],
            'total_steps': 6,  # Total number of pipeline steps
            'progress_percentage': (len(self.pipeline_state['completed_steps']) / 6) * 100,
            'errors': self.pipeline_state['errors']
        }

    def reset_pipeline(self):
        """Reset pipeline state for new execution"""
        self.pipeline_state = {
            'current_step': 'initialized',
            'completed_steps': [],
            'results': {},
            'errors': []
        }

        # Reset agents that maintain state
        self.model_builder = ModelBuildingAgent()
        self.automl_agent = AutoMLAgent()

        print("🔄 Pipeline reset successfully")

    def configure_pipeline(self, **config_updates):
        """Update pipeline configuration"""
        for section, updates in config_updates.items():
            if section in self.pipeline_config:
                self.pipeline_config[section].update(updates)
            else:
                self.pipeline_config[section] = updates

        print(f"⚙️ Pipeline configuration updated: {list(config_updates.keys())}")

    def quick_analysis(self, data_source, target_column=None, **kwargs):
        """Run a quick analysis with minimal configuration"""
        # Configure for speed
        quick_config = {
            'data_cleaning': {'aggressive_cleaning': False},
            'modeling': {
                'categories': ['traditional_ml'],  # Only basic models
                'enable_automl': False  # Skip AutoML for speed
            }
        }

        return self.execute_pipeline(
            data_source=data_source,
            target_column=target_column,
            pipeline_config=quick_config,
            **kwargs
        )

    def comprehensive_analysis(self, data_source, target_column=None, **kwargs):
        """Run a comprehensive analysis with all features enabled"""
        # Configure for completeness
        comprehensive_config = {
            'data_cleaning': {'aggressive_cleaning': True},
            'modeling': {
                'categories': ['traditional_ml', 'ensemble', 'boosting', 'deep_learning'],
                'enable_automl': True,
                'automl_time_budget': 600  # 10 minutes
            }
        }

        return self.execute_pipeline(
            data_source=data_source,
            target_column=target_column,
            pipeline_config=comprehensive_config,
            **kwargs
        )
