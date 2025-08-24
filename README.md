# 👩‍💼 Employee Attrition Predictor

A machine learning-powered web application that predicts employee attrition risk using a trained logistic regression model. Built with Streamlit for an intuitive user interface and interactive data visualization.

## 🌟 Features

### Manual Input Mode

- **Interactive UI**: User-friendly sliders and dropdowns for inputting employee data
- **Real-time Predictions**: Instant attrition risk assessment with probability scores
- **Feature Tooltips**: Helpful descriptions for each input parameter
- **Visual Results**: Interactive charts showing prediction probabilities

### Batch Processing Mode

- **CSV Upload**: Process multiple employees simultaneously
- **Batch Results**: Comprehensive results table with risk highlighting
- **Data Visualization**: Probability distribution charts for batch predictions
- **Export-Ready**: Results can be downloaded for further analysis

### Key Prediction Features

The model considers 24 carefully selected features including:

- **Demographics**: Age, Gender, Marital Status
- **Career Metrics**: Total Working Years, Years at Company, Years with Current Manager
- **Job Details**: Job Level, Job Role, Department, Monthly Income
- **Satisfaction Scores**: Job Satisfaction, Environment Satisfaction, Job Involvement
- **Work Conditions**: Overtime, Business Travel, Stock Option Level

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web application framework
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) - Logistic regression model
- **Data Processing**: [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- **Visualization**: [Plotly](https://plotly.com/) - Interactive charts and graphs
- **Data Handling**: [NumPy](https://numpy.org/) - Numerical computing
- **Model Management**: [Joblib](https://joblib.readthedocs.io/) - Model serialization

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/sreekarun/AttritionPredictor.git
   cd AttritionPredictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py

   ```
