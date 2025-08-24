import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model, scaler, and features
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load(
    "selected_features.pkl"
)  # List of columns model expects

# Streamlit app setup
st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩 Employee Attrition Predictor")
st.markdown(
    """
Estimate the likelihood of an employee leaving the company using key features.  
You can either provide details manually for one employee or upload a CSV file to analyze multiple employees at once.
"""
)

# Feature info for tooltips
feature_info = {
    "Age": "Employee's age in years.",
    "TotalWorkingYears": "Total number of years the employee has worked across all companies.",
    "YearsWithCurrManager": "Years spent with the current manager.",
    "YearsInCurrentRole": "Years spent in the current job role.",
    "YearsAtCompany": "Total years spent at this company.",
    "MonthlyIncome": "Monthly salary in USD.",
    "JobLevel": "Level of the job (1: entry level, 5: executive).",
    "StockOptionLevel": "Level of stock options awarded (0–3).",
    "JobInvolvement": "Level of employee's commitment (1: low, 4: high).",
    "JobSatisfaction": "Level of satisfaction with the job (1–4).",
    "EnvironmentSatisfaction": "Satisfaction with workplace environment (1–4).",
    "Gender": "Employee's gender.",
    "OverTime": "Whether the employee works overtime.",
    "MaritalStatus": "Marital status.",
    "BusinessTravel": "Frequency of business travel.",
    "JobRole": "Role of the employee in the company.",
    "Department": "Department where the employee works.",
}

# Sidebar input mode
mode = st.sidebar.radio("Choose Input Mode", ["Manual Input", "Upload CSV"])

# ------------------------
# Manual Input Mode
# ------------------------
if mode == "Manual Input":
    st.subheader("🔧 Manual Input: Single Employee")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30, help=feature_info["Age"])
        total_working_years = st.slider(
            "Total Working Years", 0, 40, 10, help=feature_info["TotalWorkingYears"]
        )
        years_with_curr_manager = st.slider(
            "Years with Current Manager",
            0,
            20,
            3,
            help=feature_info["YearsWithCurrManager"],
        )
        years_in_current_role = st.slider(
            "Years in Current Role", 0, 20, 3, help=feature_info["YearsInCurrentRole"]
        )
        years_at_company = st.slider(
            "Years at Company", 0, 40, 5, help=feature_info["YearsAtCompany"]
        )
        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            200000,
            5000,
            step=500,
            help=feature_info["MonthlyIncome"],
        )
        job_level = st.slider("Job Level", 1, 5, 2, help=feature_info["JobLevel"])
        stock_option_level = st.slider(
            "Stock Option Level", 0, 3, 1, help=feature_info["StockOptionLevel"]
        )

    with col2:
        job_involvement = st.slider(
            "Job Involvement", 1, 4, 3, help=feature_info["JobInvolvement"]
        )
        job_satisfaction = st.slider(
            "Job Satisfaction", 1, 4, 3, help=feature_info["JobSatisfaction"]
        )
        environment_satisfaction = st.slider(
            "Environment Satisfaction",
            1,
            4,
            3,
            help=feature_info["EnvironmentSatisfaction"],
        )
        gender = st.selectbox("Gender", ["Female", "Male"], help=feature_info["Gender"])
        overtime = st.selectbox(
            "OverTime", ["No", "Yes"], help=feature_info["OverTime"]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Divorced", "Married", "Single"],
            help=feature_info["MaritalStatus"],
        )
        business_travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            help=feature_info["BusinessTravel"],
        )
        job_role = st.selectbox(
            "Job Role",
            [
                "Laboratory Technician",
                "Sales Representative",
                "Research Director",
                "Manager",
                "Healthcare Representative",
                "Human Resources",
                "Manufacturing Director",
                "Research Scientist",
                "Sales Executive",
            ],
            help=feature_info["JobRole"],
        )
        department = st.selectbox(
            "Department",
            ["Human Resources", "Research & Development", "Sales"],
            help=feature_info["Department"],
        )

    # Create input dict
    input_dict = {
        "OverTime_Yes": 1 if overtime == "Yes" else 0,
        "TotalWorkingYears": total_working_years,
        "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
        "YearsWithCurrManager": years_with_curr_manager,
        "Age": age,
        "YearsInCurrentRole": years_in_current_role,
        "YearsAtCompany": years_at_company,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "StockOptionLevel": stock_option_level,
        "JobRole_Laboratory Technician": (
            1 if job_role == "Laboratory Technician" else 0
        ),
        "JobRole_Sales Representative": 1 if job_role == "Sales Representative" else 0,
        "BusinessTravel_Travel_Frequently": (
            1 if business_travel == "Travel_Frequently" else 0
        ),
        "Gender_Male": 1 if gender == "Male" else 0,
        "JobRole_Research Director": 1 if job_role == "Research Director" else 0,
        "JobInvolvement": job_involvement,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobRole_Manager": 1 if job_role == "Manager" else 0,
        "Department_Research & Development": (
            1 if department == "Research & Development" else 0
        ),
    }

    X_input = pd.DataFrame([input_dict])

    # Ensure all features present
    for col in selected_features:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[selected_features]

    # Scale numeric features
    numeric_cols = scaler.feature_names_in_
    X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

    # Prediction
    if st.button("🔮 Predict Attrition Risk"):
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Attrition Risk!**\n\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ **Low Attrition Risk**\n\nProbability: {probability:.2%}")

        # Probability chart
        fig = px.bar(
            x=["Low Risk", "High Risk"],
            y=[1 - probability, probability],
            labels={"x": "Prediction", "y": "Probability"},
            color=["Low Risk", "High Risk"],
            color_discrete_map={"Low Risk": "green", "High Risk": "red"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------
# CSV Upload Mode
# ------------------------
else:
    st.subheader("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with employee data", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()

        # Fill missing model columns with 0
        for col in selected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[selected_features]

        # Scale numeric columns
        numeric_cols = scaler.feature_names_in_
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        df_results = pd.DataFrame(
            {
                "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
                "Probability": probs,
            }
        )

        # Merge results with original
        results_df = pd.concat([original_df, df_results], axis=1)

        # Highlight high-risk rows
        def highlight_risk(row):
            color = (
                "background-color: salmon" if row["Prediction"] == "High Risk" else ""
            )
            return [color] * len(row)

        st.dataframe(results_df.style.apply(highlight_risk, axis=1))

        # Distribution chart
        fig = px.histogram(
            df_results,
            x="Probability",
            nbins=10,
            color="Prediction",
            title="Probability Distribution of Attrition Risk",
            color_discrete_map={"Low Risk": "green", "High Risk": "red"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="attrition_predictions.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.caption(
    "Model: Logistic Regression | Scaling: MinMaxScaler | Built by Trisha Mandal"
)
