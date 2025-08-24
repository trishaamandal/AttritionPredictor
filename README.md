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

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Manual Input Mode

1. **Select Manual Input** from the sidebar
2. **Fill in employee details** using the interactive controls:
   - Use sliders for numerical values (Age, Income, Years, etc.)
   - Use dropdowns for categorical values (Gender, Department, etc.)
3. **Click "Predict Attrition Risk"** to get results
4. **View the prediction** with probability score and visual chart

### CSV Upload Mode

1. **Select Upload CSV** from the sidebar
2. **Prepare your CSV file** with employee data columns matching the required features
3. **Upload the file** using the file uploader
4. **View batch results** with risk highlighting and probability distribution

### CSV Format Requirements

Your CSV file should include columns for the features used by the model. The application will automatically handle missing columns by setting them to default values.

Example CSV structure:
```csv
Age,TotalWorkingYears,MonthlyIncome,JobLevel,Gender,OverTime,Department,...
35,10,5000,2,Male,Yes,Sales,...
```

## 🤖 Model Information

- **Algorithm**: Logistic Regression
- **Preprocessing**: MinMaxScaler for numerical features
- **Features**: 24 selected features from employee data
- **Output**: Binary classification (High Risk / Low Risk) with probability scores
- **Performance**: Trained on employee attrition dataset with feature selection

### Model Files
- `logistic_regression_model.pkl` - Trained logistic regression model
- `scaler.pkl` - MinMaxScaler for feature preprocessing
- `selected_features.pkl` - List of 24 selected features

## 📁 File Structure

```
AttritionPredictor/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── logistic_regression_model.pkl    # Trained ML model
├── scaler.pkl                      # Feature scaler
├── selected_features.pkl           # Selected features list
└── README.md                       # Project documentation
```

## 🔧 Configuration

The application uses pre-trained models and doesn't require additional configuration. All model artifacts are included in the repository for immediate use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 Example Use Cases

- **HR Analytics**: Identify employees at risk of leaving
- **Retention Planning**: Proactive measures for high-risk employees
- **Workforce Management**: Understanding factors contributing to attrition
- **Strategic Planning**: Data-driven insights for employee retention strategies

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

## 👨‍💻 Author

**Aman Sreejesh**
- GitHub: [@sreekarun](https://github.com/sreekarun)

## 🙏 Acknowledgments

- Built using the powerful Streamlit framework
- Machine learning capabilities powered by Scikit-learn
- Interactive visualizations created with Plotly

---

*For questions, suggestions, or support, please open an issue on GitHub.*