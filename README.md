# Telco Customer Churn Prediction

A machine learning project that analyzes customer churn, compares classification models, and provides a Streamlit app for individual and batch customer scoring.

Built by **Muslih Uddin**.

## Business problem

A telecom company wants to identify customers at risk of leaving and prioritize retention outreach. This project examines historical churn patterns and builds a model that emphasizes recall, while reporting the false alarms and workload that come with it.

## Dataset

Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- 7,043 customer records and 21 original columns.
- Target: `Churn`, mapped to `No = 0` and `Yes = 1`.
- Historical churn rate: 26.54%.
- 19 input features after excluding `customerID` and the target.

The dataset contains customer demographics, subscribed services, contract details, payment methods, tenure, and billing information.

## Project workflow

1. Inspect dataset shape, data types, missing values, and duplicates.
2. Convert `TotalCharges` from text to numeric.
3. Fill the 11 blank `TotalCharges` entries with zero because all have zero tenure. This assumes cumulative billed charges have not yet accrued.
4. Exclude `customerID` from model inputs and retain every customer.
5. Explore churn by contract, tenure, monthly charges, and payment method.
6. Create an 80/20 train-test split with `stratify=y` and `random_state=42`.
7. Place numeric imputation and scaling, categorical imputation, and one-hot encoding inside a preprocessing pipeline.
8. Compare Logistic Regression, Decision Tree, and Random Forest with and without balanced class weights.
9. Select the balanced model family using five-fold training cross-validation recall, then tune it with `GridSearchCV`.
10. Evaluate the selected model once on the untouched test set at the fixed 0.50 threshold.
11. Turn the findings into business recommendations and a Streamlit scoring app.

Learned preprocessing is fitted inside each training fold. The test set does not select the model, hyperparameters, or operating threshold.

## Main findings

| Customer segment | Historical churn rate |
| --- | ---: |
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Tenure of 0 to 12 months | 47.4% |
| Tenure of 49 to 72 months | 9.5% |
| Electronic-check payment | 45.3% |
| Automatic credit-card payment | 15.2% |
| Fiber-optic internet | 41.9% |
| DSL internet | 19.0% |

These are descriptive associations. They do not prove that a particular contract, payment method, or service caused a customer to leave.

## Model comparison

The following results are mean five-fold cross-validation scores on the training partition, using a 0.50 decision threshold.

| Model | Class weight | Churn recall | Precision |
| --- | --- | ---: | ---: |
| Logistic Regression | None | 54.3% | 65.3% |
| Logistic Regression | balanced | 80.1% | 51.7% |
| Decision Tree | None | 51.2% | 63.6% |
| Decision Tree | balanced | 78.7% | 50.2% |
| Random Forest | None | 50.6% | 67.8% |
| Random Forest | balanced | 74.8% | 54.9% |

Balanced Logistic Regression achieved the highest mean recall among the balanced candidates. `GridSearchCV` selected L1 regularization with `C = 1`.

Recall matters because a false negative represents a churner the model failed to identify. However, flagging every customer would produce 100% recall with no useful prioritization. Precision and outreach workload must also guide business decisions.

## Final test results

The test set contains 1,409 customers, including 374 churners. The saved model was fitted on the 5,634 training customers and evaluated at a fixed threshold of 0.50.

| Metric | Result |
| --- | ---: |
| Churn recall | 78.9% |
| Precision | 50.4% |
| F1 score | 0.615 |
| Accuracy | 73.8% |
| ROC AUC | 0.842 |
| Average precision | 0.635 |
| Customers flagged for review | 585 of 1,409 |

### Confusion matrix

| Actual outcome | Predicted stayed | Predicted churned |
| --- | ---: | ---: |
| Stayed | 745 | 290 |
| Churned | 79 | 295 |

The model caught **295 of 374 churners** and missed 79. It also flagged 290 customers who stayed. Roughly half the flagged accounts actually churned in this historical test.

## Streamlit app

The app provides:

- Individual customer scoring using all 19 model features.
- CSV batch scoring with input validation.
- Customer records ranked by churn risk score.
- Downloadable prediction results.
- An adjustable outreach threshold.
- A confusion matrix and saved evaluation metrics.
- Cross-validation tables and charts when report files are available.
- Business recommendations based on historical patterns.

The app uses the saved preprocessing pipeline and classifier together. It does not retrain during scoring. Changing the outreach threshold changes the flags, while the displayed holdout metrics remain tied to the fixed 0.50 evaluation threshold.

The class-weighted model is uncalibrated. Its output is a risk score, not a verified probability of cancellation. Uploaded data remains in the app session, and the app code does not write uploads to disk.

## Repository files

Preserve these relative paths when uploading files to GitHub.

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entrypoint |
| `requirements.txt` | Pinned Python dependencies |
| `models/churn_pipeline.joblib` | Required saved model, preprocessing pipeline, and metadata |
| `Telco_Churn_Analysis.ipynb` | Colab-compatible analysis and model training |
| `reports/results.json` | Saved model comparisons and analysis metadata |
| `reports/figures/` | EDA, evaluation, and importance charts |
| `reports/model_comparison_cv.csv` | Cross-validation results |
| `reports/grid_search_cv.csv` | Hyperparameter search results |
| `reports/holdout_predictions.csv` | Historical test predictions for auditing |
| `reports/cleaned_customers.csv` | Cleaned historical dataset |
| `reports/permutation_importance.csv` | Predictive feature-importance results |
| `reports/segment_*.csv` | Segment counts and churn rates |
| `reports/threshold_tradeoff_training.csv` | Exploratory training threshold trade-offs |

The self-contained `app.py` generated by the notebook requires `requirements.txt` and `models/churn_pipeline.joblib`. Reports and charts extend the app when present. The original modular app from the full project ZIP additionally requires `core.py` and `data/Telco-Customer-Churn.csv`.

Do not place the model beside `app.py`. The application expects it inside the `models` folder.

## Run locally

Use Python 3.12. Open a terminal in the project folder.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in your terminal.

## Train using Google Colab

1. Open `Telco_Churn_Analysis.ipynb` in Colab.
2. Run the notebook cells in order.
3. Upload the original Telco CSV when prompted.
4. Run the model-export cell to save `models/churn_pipeline.joblib`.
5. Download the model and generated reports, preserving their folder paths.
6. Generate or download `app.py` and add `requirements.txt`.

Keep the saved model and deployment package versions consistent. The provided artifact was trained with scikit-learn 1.8.0.

## Deploy on Streamlit Community Cloud

1. Upload the project files and folders to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from your GitHub repository.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Select Python 3.12 in Advanced settings.
7. Deploy and test individual scoring, CSV uploads, and downloads.

See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Business recommendations

1. **Test optional loyalty offers for month-to-month accounts.** Compare incremental retention and contribution margin against a randomized control group.
2. **Improve first-year onboarding and support.** Test service checks, bill explanations, and follow-up calls. Track retention, resolution times, and contact costs.
3. **Investigate payment friction.** Test payment reminders and optional automatic payments. Measure failed payments and incremental retention.
4. **Review fiber-service quality and billing expectations.** Investigate support tickets, outages, and price concerns before testing targeted improvements.

The dataset does not contain retention-intervention outcomes or validated lifetime values. No revenue-saving or retention-improvement claim is made.

## Limitations

- The data represents a historical sample, with no live customer feed.
- A random holdout does not establish performance on a later time period.
- The project does not define a validated future prediction horizon.
- Segment rates and feature importance do not establish causal cancellation reasons.
- Real-world use needs pre-cancellation snapshots, later-period validation, subgroup review, outcome monitoring, and a retention experiment.

## Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit, and Google Colab.

## Author

**Muslih Uddin**  
Data Science and Machine Learning

[GitHub](https://github.com/shahmusleeh) · [LinkedIn](https://www.linkedin.com/in/musleeh-shah/)

## Data attribution

The Telco Customer Churn dataset comes from the Kaggle source linked above. Dataset rights remain with the original provider. No separate dataset license is asserted by this repository.
