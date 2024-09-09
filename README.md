# Predicting Police Call Outcomes: San Jose Police Calls for Service

**Author**: Michael Smith

## Executive Summary
This project aims to predict the outcomes of police calls in San Jose using historical call-for-service data. Various machine learning models are explored to determine whether a call will result in an arrest or citation.

## Rationale
Police departments process thousands of calls for service every day. Being able to predict the outcome of these calls, such as whether an arrest or citation will be issued, can help with resource allocation and operational efficiency.

## Research Question
Can we build a model to predict whether a police call will result in an arrest or citation?

## Data Sources
The data for this project comes from the [San Jose Police Calls For Service dataset](https://data.sanjoseca.gov/dataset/police-calls-for-service), spanning 5 years, from 2018 through 2022.

## Methodology
- **Data Cleaning**: Removed rows with missing data and unnecessary columns.
- **Feature Engineering**: Created features such as `OFFENSE_HOUR`, `OFFENSE_DAY_OF_WEEK`, and encoded categorical features like `CALLTYPE_CODE`.  Added latitude and longitude based on the address, and calculated neighborhood based on those geographical coordinates.

Consolidated the final disposition codes into 6 options:
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

| Model              | Accuracy  | Recall (macro) | Precision (macro) | F1-score (macro)  |
|--------------------|-----------|----------:|----------:|----------:|
| Baseline           | 0.962488  | 0.250000  | 0.240622  | 0.245221  |
| Logistic Regression| 0.969396  | 0.513951  | 0.691921  | 0.571131  |
| Random Forest      | 0.957442  | 0.503261  | 0.521498  | 0.507291  |
| MLP1               | 0.968847  | 0.503827  | 0.688551  | 0.562082  |
| MLP2               | 0.967549  | 0.469603  | 0.600936  | 0.500041  |
| MLP3               | 0.969923  | 0.529908  | 0.699869  | 0.586328  |
| CNN1               | 0.969595  | 0.523061  | 0.692655  | 0.579459  |
| CNN2               | 0.968741  | 0.487184  | 0.685626  | 0.535631  |
| CNN3               | 0.968970  | 0.511344  | 0.684971  | 0.565068  |

- The **Baseline model** produced an F1 score of 0.245, as expected, since it only predicts the most frequent class and has limited ability to handle minority outcomes.
- **Logistic Regression** achieved a strong performance with an F1 score of 0.571, showing effective handling of the imbalanced dataset after tuning.
- The **Random Forest model** yielded an F1 score of 0.507, performing moderately but struggling slightly with precision compared to Logistic Regression.
- Among the **Multi-Layer Perceptron models**, **MLP3** performed the best with an F1 score of 0.586, while **MLP1** and **MLP2** achieved scores of 0.562 and 0.500, respectively, indicating that MLP models can capture minority outcomes well with the right configurations.
- The **Convolutional Neural Network models** showed competitive performance, with **CNN1** achieving an F1 score of 0.579, followed by **CNN3** at 0.565 and **CNN2** at 0.536, suggesting that convolutional networks are capable of strong results on this dataset.

Overall, class imbalance continues to present challenges, but several models, especially Logistic Regression, MLP3, and CNN1, show promise in balancing precision and recall for minority classes.

## Next Steps
- Explore additional models, such as LSTMs, to capture temporal patterns in the data.
- Investigate ways to address class imbalance using techniques like SMOTE or adjusting class weights.
- Consider feature selection to reduce dimensionality and improve model performance.

## Outline of Project
- [jupyter notebook focusing on arrests and citations](arrest_only_2.ipynb)

