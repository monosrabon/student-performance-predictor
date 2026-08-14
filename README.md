# Student Performance Prediction — Streamlit Deployment

Professional Streamlit interface for the submitted Student Performance Prediction project.

## Included model
The repository includes the exported XGBoost model:

`model/student_performance_xgboost.json`

The model artifact contains **22 input features** and is used directly by the app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder.
3. Go to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Select `app.py` as the main file.
6. Deploy.

The app loads the model from `model/student_performance_xgboost.json`.

## Important project limitation

The model uses first- and second-semester academic variables. Therefore, it should not be described as a purely enrollment-time early-warning model.

The underlying project reported 74.04% final validation accuracy and particularly weak recall for the Enrolled class. Predictions should be treated as decision support rather than definitive student-status decisions.
