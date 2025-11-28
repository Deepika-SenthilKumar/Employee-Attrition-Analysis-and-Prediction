OBJECTIVE:
	This project focuses on building a machine learning solution to predict employee attrition and deploying the insights through an interactive Streamlit dashboard for HR analysis.

1.DATASET AND PREPROCESSING:
	The project utilized an employee dataset (Employee-Attrition.csv) containing various attributes:
		(a)Numeric Features: Age, MonthlyIncome, YearsAtCompany, YearsInCurrentRole, etc.
		(b)Categorical Features: JobRole, Department, BusinessTravel, MaritalStatus, Gender, OverTime, etc.
		(c)Target Variable: Attrition (Yes/No or 1/0).

	Data Preprocessing Steps
		(a)Column Removal: Unnecessary columns like EmployeeCount, Over18, and StandardHours were dropped as they 		had little value 			for the prediction model.
		(b)Categorical Encoding: The LabelEncoder was applied to convert categorical features (e.g., JobRole, Department) 		into a 				numeric format suitable for machine learning models.
		(c)Feature Scaling: The StandardScaler was used on all numeric features to normalize the data, ensuring all features 						contribute equally to the model training process.
		(d)Handling Imbalance: To prevent model bias towards the non-attrition class, the RandomOverSampler technique 		was employed 			to balance the number of Yes and No instances in the target variable.

2. FEATURE ENGINEERING:
	Two new features were created to potentially boost model performance:
		(a)TenurePerJobLevel: Calculated as YearsAtCompany / JobLevel to measure time spent at the company relative to 		seniority.
		(b)PromotionLag: Calculated as YearsSinceLastPromotion / YearsAtCompany to quantify the gap between 		promotions and overall 			tenure.

3. MODEL TRAINING AND EVALUATION:
	Two different machine learning models were trained and evaluated:

	(A)Logistic Regression (Baseline)
		(i)This model served as the initial baseline to compare against more complex models.
		(ii)Evaluation Metrics: Accuracy, Recall, Precision, F1-score, and AUROC (Area Under the Receiver Operating 		Characteristic 			curve) were calculated.

	(B)Random Forest Classifier (Main Model)
		(i)Chosen as the primary predictive model due to its ability to handle complex relationships and provide feature 							importances.
		(ii)Feature Importance: Calculated to identify which factors (e.g., OverTime, MonthlyIncome, Age) have the greatest 						influence on an employee's decision to leave.
		(iii)Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC.
		(iv)Deployment Preparation: The trained Random Forest model was saved using the pickle library for later use in the 						Streamlit dashboard for real-time predictions.

4. STREAMLIT DASHBOARD:
	An interactive web dashboard was developed using Streamlit to visualize data insights and host the prediction model.
  Dashboard Tabs:
    (a) Predict Attrition
        (i)Interactive Form: Allows HR to input a new employee's details (age, income, job role, etc.).
        (ii)Prediction Pipeline: The dashboard handles categorical encoding and numeric scaling internally before 			feeding the 			data to the saved Random Forest model.
        (iii)Attrition Risk: Outputs the employee's attrition risk in percentage.
  
     (b)Employee Insights: Displays three critical tables side-by-side using st.columns:
         (i)High-Risk Employees
         (ii)High Job Satisfaction Employees
        (iii)High Performance Employees
