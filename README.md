# Heart Attack Prediction Project  

This project focuses on predicting heart attack using various machine learning models, considering various health risk factors.

## Key Objectives : 

The goal of this project was to build a supervised machine learning pipeline to predict the likelihood of a heart attack based on clinical and lifestyle factors. The main objectives are:  
- Explore patterns and correlations between health indicators (age, cholesterol, blood pressure, chest pain type, etc.) and heart attack risk  
- Train and compare multiple classification models to identify which best captures risk patterns  
- Evaluate predictive performance with a focus on precision, recall, and decision intelligence  
- Interpret results both technically (metrics, plots) and medically (which features matter for heart attack risk)  

## Methodology:  

1. **Data Preparation**  
   - Cleaned and filtered patient records  
   - Encoded categorical factors like chest pain type and exercise-induced angina  
   - Normalized continuous variables such as cholesterol and maximum heart rate  
   - Applied data versioning to keep preprocessing steps traceable  

2. **Descriptive Analytics**  
   - Target distribution showed **class imbalance**: fewer positive cases of heart attack compared to negatives  
   - Correlation analysis revealed strong links between features like chest pain type, ST depression, and heart attack occurrence  
   - Visualization showed lifestyle and exercise-related factors had more predictive signal than static measures like resting blood pressure  

3. **Model Training and Tuning**  
   - Models tested: **Logistic Regression, Decision Tree, Random Forest, Support Vector Machine**  
   - Cross-validation used for reliability  
   - Hyperparameter tuning with GridSearchCV refined performance  
   - Metrics: Accuracy, Precision, Recall, and F1 score  

## Model Pipeline:  
1. Load and version dataset  
2. Preprocessing (scaling, encoding, cleaning)  
3. Exploratory analysis (target imbalance, correlations, feature distributions)  
4. Train baseline models  
5. Apply cross-validation  
6. Perform hyperparameter tuning  
7. Evaluate and visualize results (confusion matrices, ROC curves, feature importance)  

## Challenges Addressed:  

- **Severe class imbalance**: Models favored predicting non-heart attack cases, leading to low recall  
- **Precision–recall trade-off**: High precision meant very few false positives, but too many missed positives  
- **Feature interpretability**: Needed to go beyond metrics and explain medically why certain features drive predictions  
- **Practical application**: Building a foundation for predictive systems that clinicians can trust and extend  

## Results:

While models avoided false alarms (high precision), they struggled to identify at-risk patients (low recall).  
### Model Metrics -   

| Model                   | Accuracy | Precision | Recall  | F1 Score |
|--------------------------|----------|-----------|---------|----------|
| Logistic Regression      | 0.651    | 0.944     | 0.027   | 0.053    |
| Decision Tree            | 0.655    | 0.962     | 0.040   | 0.076    |
| Random Forest            | 0.655    | 0.962     | 0.040   | 0.076    |
| Support Vector Machine   | 0.655    | 0.962     | 0.040   | 0.076    |  

### Cross-Validation Accuracy  - 
- Logistic Regression: **0.649**  
- Decision Tree: **0.651**  
- Random Forest: **0.651**  
- Support Vector Machine: **0.651**  

### Best Hyperparameters  - 
- Logistic Regression: `C = 10`  
- Decision Tree: `max_depth = 7`  
- Random Forest: `n_estimators = 50`  
- Support Vector Machine: `C = 0.1`  

### Confusion Matrices  - 
Across all models, **false negatives dominated** → most actual heart attack cases were missed.  
- Logistic Regression and SVM predicted almost all patients as “no heart attack”  
- Decision Tree and Random Forest caught slightly more positives, but recall was still weak  

While models avoided false alarms (high precision), they struggled to identify at-risk patients (low recall).  

### ROC Curves  -
ROC curves confirmed weak separability between classes:  
- AUC scores hovered near 0.5, close to random guessing  
- Logistic Regression and SVM performed slightly better than Decision Tree and Random Forest, but not enough to be clinically actionable
  
### Medical Interpretation of Features  -
Even though predictive performance was limited, feature analysis revealed meaningful insights:  
- **Chest Pain Type (cp)**: Strongest predictor. Patients with atypical angina or asymptomatic chest pain had significantly higher risk.  
- **ST Depression (oldpeak)**: Indicates exercise-induced stress on the heart; higher values correlated with increased heart attack likelihood.  
- **Exercise-Induced Angina (exang)**: Patients experiencing angina during physical activity were flagged more often as high risk.  
- **Age**: Older patients showed higher probability of heart attack, as expected clinically.  
- **Cholesterol and Resting BP**: Surprisingly, these were weaker predictors compared to exercise and ECG-related factors, suggesting that static measures may not capture short-term risk effectively.

## Impact:  
- Built a disease prediction pipeline connecting descriptive health analytics to predictive modeling  
- Demonstrated the limitations of relying only on accuracy in imbalanced medical datasets  
- Surfaced clinically relevant risk factors that align with medical research (chest pain type, ST depression, exercise-induced angina, age)  
- Established a reproducible framework that can be extended with class balancing (SMOTE, weights), ensemble models, or deep learning for improved recall  

## Technology and Tools:  
- **Python** – core development  
- **Pandas, NumPy** – preprocessing and data filtering  
- **Matplotlib, Seaborn** – distribution plots, correlation heatmaps, confusion matrices, ROC curves  
- **Scikit-learn** – classification models, cross-validation, hyperparameter tuning  
- **Jupyter Notebook** – iterative analysis and documentation  

