# Predicting Police Call Outcomes: San Jose Police Calls for Service

**Author**: Michael Smith

## Executive Summary
This project aims to predict the outcomes of police calls in San Jose using historical call-for-service data. Various machine learning models are explored to determine whether a call will result in an arrest or citation.

## Rationale
Police departments process thousands of calls for service every day. Being able to predict the outcome of these calls, such as whether an arrest or citation will be issued, can help with resource allocation and operational efficiency.

## Research Question
Can we build a model to predict whether a police call will result in an arrest or citation?

## Data Sources
The data for this project comes from the [San Jose Police Calls For Service dataset](https://data.sanjoseca.gov/dataset/police-calls-for-service), spanning multiple years.

## Methodology
- **Data Cleaning**: Removed rows with missing data and unnecessary columns.
- **Feature Engineering**: Created features such as `OFFENSE_HOUR`, `OFFENSE_DAY_OF_WEEK`, and encoded categorical features like `CALLTYPE_CODE`. Consolidated the final disposition codes into 6 options:
  - A: Arrest Made
  - B: Arrest by Warrant
  - C: Criminal Citation
  - D: Traffic Citation Issued, Hazardous
  - E: Traffic Citation Issued, Non-Hazardous
  - Other
- **Modeling**: Several machine learning models were applied:
  - Baseline model (predicting the most frequent class)
  - Logistic Regression
  - Random Forest
  - Multi-Layer Perceptron (MLP)
- **Evaluation Metrics**: Models were evaluated using accuracy, precision, recall, and F1-score. Hyperparameter tuning was performed using Grid Search.

## Results
- The **Baseline model** achieved an accuracy of 0.89, as expected, since it only predicts the most frequent class.
- **Logistic Regression** performed well with an accuracy of 0.91 after hyperparameter tuning.
- The **MLP neural network** achieved an accuracy of 0.91, suggesting that more complex models may improve performance slightly.
- **Random Forest** performed similarly to others with an accuracy of 0.91 after hyperparameter tuning.
- Class imbalance remains a challenge, especially for minority classes like arrest outcomes.

## Next Steps
- Explore additional models, such as LSTMs, to capture temporal patterns in the data.
- Investigate ways to address class imbalance using techniques like SMOTE or adjusting class weights.
- Consider feature selection to reduce dimensionality and improve model performance.

## Outline of Project
- [jupyter notebook focusing on arrests and citations](arrest_only.ipynb)

