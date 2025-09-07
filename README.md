# 🚀 End-to-End Data Science Pipeline

A comprehensive, automated data science pipeline that orchestrates the entire machine learning workflow from data loading to model optimization using multiple specialized agents.

## 🌟 Features

- **Automated Pipeline**: Complete end-to-end workflow with minimal manual intervention
- **Multi-Agent Architecture**: Specialized agents for each pipeline stage
- **Domain Intelligence**: Automatic domain detection and expert recommendations
- **AutoML Integration**: Automated model optimization and hyperparameter tuning
- **Comprehensive Analysis**: Detailed EDA, cleaning, and reporting
- **Flexible Configuration**: Customizable pipeline settings for different use cases
- **Export Capabilities**: Results export in JSON/YAML formats

## 🏗️ Architecture

The pipeline consists of specialized agents orchestrated by a `SupervisorAgent`:

SupervisorAgent (Main Orchestrator)
├── DataLoaderAgent     → Data ingestion from various sources
├── DataCleaningAgent   → Data preprocessing and quality improvement
├── EDAAgent           → Exploratory data analysis and insights
├── DomainExpertAgent  → Domain-specific recommendations
├── ModelBuildingAgent → Traditional ML model training
└── AutoMLAgent        → Automated model optimization

## 🚦 Pipeline Steps

1. **📁 Data Loading** - Ingests data from CSV, JSON, and other formats
2. **🧹 Data Cleaning** - Handles missing values, duplicates, and outliers
3. **📊 Exploratory Analysis** - Generates insights and visualizations
4. **🎓 Domain Analysis** - Provides domain-specific recommendations
5. **🤖 Model Building** - Trains multiple ML models
6. **🔧 AutoML Optimization** - Optimizes model performance automatically

## 🛠️ Installation

### Prerequisites
python >= 3.8
pandas
numpy
scikit-learn
plotly
requests
openpyxl
seaborn
matplotlib


### Setup
git clone https://github.com/Giri530/Data-Scientist-Multiagent-System.git
cd Data-Scientist-Multiagent-System
pip install -r requirements.txt

## 🚀 Quick Start

### Basic Usage

from supervisor_agent import SupervisorAgent

# Initialize the pipeline
pipeline = SupervisorAgent()

# Run complete analysis
results = pipeline.execute_pipeline(
    data_source='path/to/your/data.csv',
    target_column='target_variable',
    domain='finance'  # Optional domain hint
)

# Print summary
print(pipeline.generate_pipeline_summary(results))

### Quick Analysis (Fast)

# For rapid prototyping
results = pipeline.quick_analysis(
    data_source='data.csv',
    target_column='target'
)

### Comprehensive Analysis (Complete)

# For thorough analysis with all features
results = pipeline.comprehensive_analysis(
    data_source='data.csv',
    target_column='target'
)


## ⚙️ Configuration

### Pipeline Configuration

config = {
    'data_cleaning': {
        'aggressive_cleaning': True,
        'handle_outliers': True
    },
    'modeling': {
        'categories': ['traditional_ml', 'ensemble', 'boosting'],
        'enable_automl': True,
        'automl_time_budget': 600  # seconds
    },
    'output': {
        'generate_visualizations': True,
        'create_report': True
    }
}

results = pipeline.execute_pipeline(
    data_source='data.csv',
    target_column='target',
    pipeline_config=config
)


### Dynamic Configuration

# Update configuration on-the-fly
pipeline.configure_pipeline(
    modeling={'enable_automl': False},
    data_cleaning={'aggressive_cleaning': True}
)

## 📊 Example Output

🚀 Starting End-to-End Data Science Pipeline...
============================================================
📁 Step 1: Loading data...
✅ Data loaded successfully. Shape: (1000, 15)
   Columns: feature1, feature2, feature3, target, category...

🧹 Step 2: Cleaning data...
✅ Data cleaned successfully. New shape: (987, 15)
   Removed 13 duplicates
   Handled 3 columns with missing values

📊 Step 3: Performing EDA...
✅ EDA completed successfully
   Found 8 key insights

🎓 Step 4: Getting domain insights...
✅ Domain analysis completed
   Detected domain: finance (confidence: 0.87)
   Generated 12 recommendations

🤖 Step 5: Building models for target 'price'...
✅ Models built successfully
   Problem type: regression
   Best model: RandomForestRegressor

🔧 Step 5b: AutoML optimization...
✅ AutoML optimization completed
   Best optimized model: XGBRegressor (score: 0.8947)

📈 Step 6: Generating comprehensive report...
✅ Report generated successfully

🎉 Pipeline completed successfully!

## 📈 Results Structure

The pipeline returns comprehensive results:
{
    'status': 'success',
    'pipeline_results': {...},  # Detailed results from each step
    'final_report': {
        'executive_summary': [...],
        'data_overview': {...},
        'exploratory_analysis': {...},
        'domain_insights': {...},
        'modeling_results': {...},
        'recommendations': [...]
    },
    'data_shape': (987, 15),
    'target_column': 'target',
    'best_model': 'RandomForestRegressor',
    'automl_best': {
        'name': 'XGBRegressor',
        'score': 0.8947,
        'best_params': {...}
    }
}

## 🔧 Advanced Usage

### Pipeline Status Monitoring
# Check pipeline progress
status = pipeline.get_pipeline_status()
print(f"Progress: {status['progress_percentage']:.1f}%")
print(f"Current step: {status['current_step']}")

### Results Export
# Export results to JSON
json_export = pipeline.export_results(results, 'json', 'results.json')

# Export to YAML
yaml_export = pipeline.export_results(results, 'yaml', 'results.yaml')


### Pipeline Reset
# Reset for new analysis
pipeline.reset_pipeline()

## 🎯 Use Cases

- **Business Analytics**: Sales forecasting, customer segmentation
- **Financial Modeling**: Risk assessment, fraud detection
- **Healthcare**: Patient outcome prediction, diagnostic assistance  
- **Marketing**: Campaign optimization, conversion prediction
- **Operations**: Demand forecasting, quality control

## 📋 Supported Data Formats

- CSV files
- JSON files
- Excel files (with additional dependencies)
- Pandas DataFrames
- SQL databases (with additional configuration)

## 🤖 Model Types

### Traditional ML
- Linear/Logistic Regression
- Random Forest
- Support Vector Machines
- Gradient Boosting

### Ensemble Methods
- Voting Classifiers
- Bagging Ensembles
- Stacking Models

### AutoML Optimization
- Hyperparameter tuning
- Feature selection
- Model architecture optimization
- Cross-validation strategies

## 📊 Domain Support

The pipeline provides specialized insights for:

- **Finance**: Risk metrics, financial ratios
- **Healthcare**: Clinical indicators, patient outcomes
- **Marketing**: Conversion funnels, customer lifetime value
- **Manufacturing**: Quality control, predictive maintenance
- **Retail**: Inventory optimization, demand forecasting

## 🛡️ Error Handling

The pipeline includes robust error handling:

- Graceful failure recovery
- Detailed error reporting
- Partial results preservation
- Step-by-step error tracking
- 
# Handle pipeline errors
if results['status'] == 'error':
    print(f"Pipeline failed at: {results['failed_step']}")
    print(f"Error: {results['error']}")
    print(f"Partial results available: {results['partial_results'].keys()}")

## 🔍 Monitoring & Debugging
### Pipeline State

# Get current pipeline state
state = pipeline.pipeline_state
print(f"Completed steps: {state['completed_steps']}")
print(f"Current step: {state['current_step']}")
print(f"Errors: {state['errors']}")

### Data Quality Assessment
# Get data quality score
final_report = results['final_report']
quality_score = final_report['technical_details']['data_quality_score']
print(f"Data Quality Score: {quality_score:.3f}")
