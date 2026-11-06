from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/churn_pipeline.joblib")


try:
    artifact = load_model()
except FileNotFoundError:
    st.error(
        "Model file missing. Save models/churn_pipeline.joblib "
        "from your notebook first."
    )
    st.stop()

model = artifact["pipeline"]
features = artifact["features"]
categories = artifact["categories"]

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

internet_addons = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def prepare_inputs(data):
    """Apply the same deterministic cleaning used in the notebook."""
    missing = sorted(set(features) - set(data.columns))

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    if data.empty:
        raise ValueError("The file contains no customer rows.")

    cleaned = data[features].copy()

    for column in numeric_columns:
        original = cleaned[column]
        blank = original.isna() | original.astype(str).str.strip().eq("")

        cleaned[column] = pd.to_numeric(original, errors="coerce")

        invalid = cleaned[column].isna() & ~blank

        if invalid.any():
            raise ValueError(f"{column} contains invalid numbers.")

        if (
            np.isinf(cleaned[column]).any()
            or cleaned[column].dropna().lt(0).any()
        ):
            raise ValueError(
                f"{column} must contain finite, non-negative values."
            )

    # Match the notebook's zero-tenure billing assumption.
    zero_tenure_blank = (
        cleaned["tenure"].eq(0)
        & cleaned["TotalCharges"].isna()
    )
    cleaned.loc[zero_tenure_blank, "TotalCharges"] = 0.0

    if cleaned[["tenure", "MonthlyCharges"]].isna().any().any():
        raise ValueError("Tenure and MonthlyCharges are required.")

    if (cleaned["tenure"] % 1 != 0).any():
        raise ValueError("Tenure must contain whole months.")

    for column, allowed in categories.items():
        if column == "SeniorCitizen":
            cleaned[column] = cleaned[column].map(
                lambda value: (
                    str(int(float(value)))
                    if str(value).strip() in ["0", "1", "0.0", "1.0"]
                    else value
                )
            )

        cleaned[column] = cleaned[column].map(
            lambda value: str(value).strip()
            if pd.notna(value) else value
        )

        if not cleaned[column].isin(allowed).all():
            raise ValueError(
                f"{column}: allowed values are {', '.join(allowed)}"
            )

    no_internet = cleaned["InternetService"].eq("No")

    for column in internet_addons:
        inconsistent = (
            no_internet & ~cleaned[column].eq("No internet service")
        ) | (
            ~no_internet & cleaned[column].eq("No internet service")
        )

        if inconsistent.any():
            raise ValueError(
                f"{column} is inconsistent with InternetService."
            )

    no_phone = cleaned["PhoneService"].eq("No")

    inconsistent_phone = (
        no_phone & ~cleaned["MultipleLines"].eq("No phone service")
    ) | (
        ~no_phone & cleaned["MultipleLines"].eq("No phone service")
    )

    if inconsistent_phone.any():
        raise ValueError(
            "MultipleLines is inconsistent with PhoneService."
        )

    return cleaned


def show_range_warnings(data):
    for column in numeric_columns:
        lower, upper = artifact["training_ranges"][column]

        outside = (
            (data[column] < lower)
            | (data[column] > upper)
        )

        if outside.any():
            st.warning(
                f"{column}: {int(outside.sum())} values fall outside "
                f"the training range {lower:g} to {upper:g}."
            )


with st.sidebar:
    st.title("📡 Retention Desk")

    threshold = st.slider(
        "Outreach score threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(artifact["threshold"]),
        step=0.05,
    )

    st.caption(
        "Lower thresholds flag more customers and usually "
        "increase recall."
    )

    st.write("Model:", artifact["model_name"])
    st.caption("Historical sample data. No live customer feed.")


st.title("Telco Customer Churn Prediction")

st.write(
    "Score customer accounts and prioritize retention outreach."
)

metrics = artifact["metrics"]

columns = st.columns(4)
columns[0].metric("Holdout recall", f"{metrics['recall']:.1%}")
columns[1].metric("Holdout precision", f"{metrics['precision']:.1%}")
columns[2].metric("Churners caught", metrics["tp"])
columns[3].metric("Missed churners", metrics["fn"])

st.caption(
    "Holdout metrics use the fixed evaluation threshold of 0.50. "
    "Changing the sidebar threshold changes scoring flags only. "
    "Scores are uncalibrated and are not verified churn probabilities."
)

single_tab, batch_tab, model_tab, business_tab = st.tabs(
    [
        "Individual prediction",
        "Batch prediction",
        "Model results",
        "Business recommendations",
    ]
)


with single_tab:
    st.subheader("Customer details")

    account, services, household = st.columns(3)
    customer = {}

    with account:
        customer["tenure"] = st.number_input(
            "Tenure in months", 0, 600, 12
        )

        customer["MonthlyCharges"] = st.number_input(
            "Monthly charges", 0.0, 100000.0, 75.0
        )

        customer["TotalCharges"] = st.number_input(
            "Total charges", 0.0, 10000000.0, 900.0
        )

        for column in ["Contract", "PaymentMethod", "PaperlessBilling"]:
            customer[column] = st.selectbox(
                column, categories[column]
            )

    with services:
        customer["PhoneService"] = st.selectbox(
            "Phone service", ["Yes", "No"]
        )

        if customer["PhoneService"] == "No":
            customer["MultipleLines"] = "No phone service"
        else:
            customer["MultipleLines"] = st.selectbox(
                "Multiple lines", ["No", "Yes"]
            )

        customer["InternetService"] = st.selectbox(
            "Internet service", categories["InternetService"]
        )

        for column in internet_addons:
            if customer["InternetService"] == "No":
                customer[column] = "No internet service"
            else:
                customer[column] = st.selectbox(
                    column, ["No", "Yes"]
                )

    with household:
        for column in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            customer[column] = st.selectbox(
                column, categories[column]
            )

    prepared = prepare_inputs(pd.DataFrame([customer]))
    score = float(model.predict_proba(prepared)[0, 1])
    flagged = score >= threshold

    st.divider()
    st.metric("Historical churn risk score", f"{score:.3f}")
    st.progress(score)

    if flagged:
        st.warning("Flagged for retention review.")
    else:
        st.success("Below the selected outreach threshold.")

    show_range_warnings(prepared)

    result = pd.DataFrame(
        [{
            **customer,
            "churn_score": score,
            "flagged_for_review": flagged,
            "threshold": threshold,
        }]
    )

    st.download_button(
        "Download prediction",
        result.to_csv(index=False).encode("utf-8"),
        "customer_prediction.csv",
        "text/csv",
    )


with batch_tab:
    st.subheader("Upload customer records")

    st.write(
        "Your CSV must contain the same 19 feature columns "
        "used during training. customerID is optional."
    )

    template = pd.DataFrame(columns=features)

    st.download_button(
        "Download CSV column template",
        template.to_csv(index=False).encode("utf-8"),
        "customer_template.csv",
        "text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)

            if len(uploaded_data) > 50000:
                raise ValueError(
                    "Upload no more than 50,000 rows per file."
                )

            prepared_batch = prepare_inputs(uploaded_data)

            scores = model.predict_proba(prepared_batch)[:, 1]

            output = uploaded_data.drop(
                columns=[
                    "Churn",
                    "churn_score",
                    "flagged_for_review",
                    "threshold",
                ],
                errors="ignore",
            ).copy()

            output["churn_score"] = scores
            output["flagged_for_review"] = scores >= threshold
            output["threshold"] = threshold

            output = output.sort_values(
                "churn_score", ascending=False, kind="stable"
            )

            show_range_warnings(prepared_batch)

            first, second = st.columns(2)
            first.metric("Customers scored", len(output))
            second.metric(
                "Flagged for review",
                int(output["flagged_for_review"].sum()),
            )

            st.dataframe(
                output, hide_index=True, width="stretch"
            )

            st.download_button(
                "Download ranked predictions",
                output.to_csv(index=False).encode("utf-8"),
                "ranked_predictions.csv",
                "text/csv",
            )

        except (
            ValueError,
            KeyError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.error(str(error))


with model_tab:
    st.subheader("Holdout confusion matrix")

    matrix = pd.DataFrame(
        artifact["test_confusion_matrix"],
        index=["Actual stayed", "Actual churned"],
        columns=["Predicted stayed", "Predicted churned"],
    )

    st.dataframe(matrix, width="stretch")

    st.write(
        f"The model caught {metrics['tp']} churners, "
        f"missed {metrics['fn']}, and raised "
        f"{metrics['fp']} false alarms."
    )

    st.json(artifact["best_params"])

    report_path = ROOT / "reports/results.json"

    if report_path.exists():
        report = pd.read_json(report_path, typ="series")

        if "cv_comparison" in report:
            st.subheader("Training cross-validation comparison")
            st.dataframe(
                pd.DataFrame(report["cv_comparison"]).round(4),
                hide_index=True,
                width="stretch",
            )

    for filename in ["evaluation.png", "importance.png"]:
        image_path = ROOT / "reports/figures" / filename

        if image_path.exists():
            st.image(str(image_path))


with business_tab:
    st.subheader("Recommended retention experiments")

    recommendations = [
        (
            "Review month-to-month accounts",
            "Test optional loyalty benefits or contract offers. "
            "Measure incremental retention and net margin.",
        ),
        (
            "Improve first-year support",
            "Test onboarding calls, service checks and bill "
            "explanations. Track retention and support resolution.",
        ),
        (
            "Investigate payment friction",
            "Test payment reminders and optional automatic "
            "payments. Track failed payments and retention.",
        ),
        (
            "Review fiber service and billing expectations",
            "Investigate reliability, support tickets and "
            "price concerns. Test targeted improvements.",
        ),
    ]

    for title, description in recommendations:
        st.subheader(title)
        st.write(description)

    st.caption(
        "Historical associations suggest experiments. "
        "They do not prove why a customer cancelled."
    )

    eda_path = ROOT / "reports/figures/eda.png"

    if eda_path.exists():
        st.image(str(eda_path))
