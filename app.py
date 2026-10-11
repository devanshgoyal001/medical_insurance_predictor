"""
Medical Insurance Charges — Analysis & Prediction Dashboard
============================================================
A Streamlit dashboard that explores the relationship between demographic /
lifestyle features and recorded medical charges, and provides a simple
regression-based cost estimator.

Dataset: Kaggle "Medical Cost Personal Datasets" (1 338 records, 7 columns).
Source : https://www.kaggle.com/datasets/mirichoi0218/insurance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Medical Insurance Charges Analysis",
    page_icon="🏥",
    layout="wide",
)

# Load data
DATA_PATH = Path(__file__).resolve().parent / "insurance.csv"


@st.cache_data
def load_data():
    """Load the insurance CSV and add derived columns."""
    data = pd.read_csv(DATA_PATH)

    # cut() places numbers into groups. Age groups include their upper limit.
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-100"]
    data["age_band"] = pd.cut(data["age"], bins=bins, labels=labels)

    # right=False includes the lower limit: BMI 25 belongs to Overweight.
    bmi_bins = [0, 18.5, 25, 30, float("inf")]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    data["bmi_category"] = pd.cut(data["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)

    return data


data = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

regions = sorted(data["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region", options=regions, default=regions
)

smoking_filter = st.sidebar.selectbox(
    "Smoking status", ["All", "yes", "no"]
)

age_range = st.sidebar.slider(
    "Age range",
    int(data["age"].min()),
    int(data["age"].max()),
    (int(data["age"].min()), int(data["age"].max())),
)

bmi_range = st.sidebar.slider(
    "BMI range",
    float(data["bmi"].min()),
    float(data["bmi"].max()),
    (float(data["bmi"].min()), float(data["bmi"].max())),
)

# Keep matching rows one filter at a time.
filtered_data = data[data["region"].isin(selected_regions)]
filtered_data = filtered_data[filtered_data["age"].between(age_range[0], age_range[1])]
filtered_data = filtered_data[filtered_data["bmi"].between(bmi_range[0], bmi_range[1])]
if smoking_filter != "All":
    filtered_data = filtered_data[filtered_data["smoker"] == smoking_filter]

st.sidebar.markdown(f"**Showing {len(filtered_data):,} of {len(data):,} records**")

# Tabs
overview_tab, comparison_tab, quality_tab, prediction_tab = st.tabs([
    "📊 Overview",
    "👥 Group Comparisons",
    "🔎 Data Quality",
    "🤖 Prediction & Evaluation",
])

# TAB 1 — Overview
with overview_tab:
    st.header("Dataset Overview")

    if filtered_data.empty:
        st.warning("No records match the current filters.")
    else:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Records", f"{len(filtered_data):,}")
        col2.metric("Avg Charges", f"${filtered_data['charges'].mean():,.0f}")
        col3.metric("Median Charges", f"${filtered_data['charges'].median():,.0f}")
        col4.metric("Max Charges", f"${filtered_data['charges'].max():,.0f}")

        st.markdown("---")

        # Distribution of charges
        left, right = st.columns(2)

        with left:
            st.subheader("Distribution of Charges")
            figure, chart = plt.subplots(figsize=(7, 4))
            chart.hist(
                filtered_data["charges"], bins=40, color="#4C72B0",
                edgecolor="white", alpha=0.85,
            )
            chart.axvline(
                filtered_data["charges"].mean(), color="#C44E52",
                linestyle="--", linewidth=1.5, label="Mean",
            )
            chart.axvline(
                filtered_data["charges"].median(), color="#55A868",
                linestyle="--", linewidth=1.5, label="Median",
            )
            chart.set_xlabel("Charges ($)")
            chart.set_ylabel("Frequency")
            chart.legend()
            chart.set_title("Charge Distribution (filtered)")
            plt.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        with right:
            st.subheader("Charges by Smoking Status")
            figure, chart = plt.subplots(figsize=(7, 4))
            sns.boxplot(
                data=filtered_data, x="smoker", y="charges",
                color="#4C72B0", ax=chart,
            )
            chart.set_xlabel("Smoker")
            chart.set_ylabel("Charges ($)")
            chart.set_title("Smoker vs Non-Smoker Charges")
            plt.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        # Scatter: age vs charges
        st.subheader("Age vs Charges (coloured by smoking status)")
        figure, chart = plt.subplots(figsize=(10, 5))
        colors = {"yes": "#C44E52", "no": "#4C72B0"}
        for status, group_data in filtered_data.groupby("smoker"):
            chart.scatter(
                group_data["age"], group_data["charges"], alpha=0.55, s=25,
                label=f"Smoker: {status}", color=colors.get(status, "grey"),
            )
        chart.set_xlabel("Age")
        chart.set_ylabel("Charges ($)")
        chart.legend()
        chart.set_title("Age vs Charges")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # BMI vs charges
        st.subheader("BMI vs Charges (coloured by smoking status)")
        figure, chart = plt.subplots(figsize=(10, 5))
        for status, group_data in filtered_data.groupby("smoker"):
            chart.scatter(
                group_data["bmi"], group_data["charges"], alpha=0.55, s=25,
                label=f"Smoker: {status}", color=colors.get(status, "grey"),
            )
        chart.set_xlabel("BMI")
        chart.set_ylabel("Charges ($)")
        chart.legend()
        chart.set_title("BMI vs Charges")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # Percentile table
        st.subheader("Charge Percentiles")
        percentiles = filtered_data["charges"].quantile(
            [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).reset_index()
        percentiles.columns = ["Percentile", "Charges ($)"]
        # Keep charges numeric and round to two decimal places.
        percentiles["Percentile"] = percentiles["Percentile"] * 100
        percentiles = percentiles.rename(columns={"Percentile": "Percentile (%)"})
        percentiles["Charges ($)"] = percentiles["Charges ($)"].round(2)
        st.dataframe(percentiles, hide_index=True, width="stretch")


# TAB 2 — Group Comparisons
with comparison_tab:
    st.header("Group Comparisons")

    if filtered_data.empty:
        st.warning("No records match the current filters.")
    else:
        # By Smoking Status
        st.subheader("Charges by Smoking Status")
        smoking_stats = filtered_data.groupby("smoker")["charges"]
        smoking_stats = smoking_stats.agg(["count", "mean", "median", "std", "min", "max"])
        smoking_stats = smoking_stats.round(2)
        smoking_stats = smoking_stats.reset_index()
        smoking_stats.columns = [
            "Smoker", "Count", "Mean ($)", "Median ($)",
            "Std Dev ($)", "Min ($)", "Max ($)",
        ]
        st.dataframe(smoking_stats, hide_index=True, width="stretch")

        if "yes" in filtered_data["smoker"].values and "no" in filtered_data["smoker"].values:
            smoker_average = filtered_data.loc[filtered_data["smoker"] == "yes", "charges"].mean()
            nonsmoker_average = filtered_data.loc[filtered_data["smoker"] == "no", "charges"].mean()
            ratio = smoker_average / nonsmoker_average if nonsmoker_average else 0
            st.info(
                f"💡 On average, smokers are charged **{ratio:.1f}×** as much "
                f"as non-smokers in the filtered data."
            )

        # By Region
        st.subheader("Charges by Region")
        region_stats = filtered_data.groupby("region")["charges"]
        region_stats = region_stats.agg(["count", "mean", "median"])
        region_stats = region_stats.round(2)
        region_stats = region_stats.sort_values("mean", ascending=False)
        region_stats = region_stats.reset_index()
        region_stats.columns = ["Region", "Count", "Mean ($)", "Median ($)"]
        st.dataframe(region_stats, hide_index=True, width="stretch")

        figure, chart = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=region_stats, x="Region", y="Mean ($)",
            color="#4C72B0", ax=chart,
        )
        chart.set_ylabel("Mean Charges ($)")
        chart.set_title("Average Charges by Region")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # By Age Band
        st.subheader("Charges by Age Band")
        age_stats = filtered_data.groupby("age_band", observed=True)["charges"]
        age_stats = age_stats.agg(["count", "mean", "median"])
        age_stats = age_stats.round(2)
        age_stats = age_stats.reset_index()
        age_stats.columns = ["Age Band", "Count", "Mean ($)", "Median ($)"]
        st.dataframe(age_stats, hide_index=True, width="stretch")

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(
            data=age_stats, x="Age Band", y="Mean ($)",
            color="#4C72B0", ax=axes[0],
        )
        axes[0].set_title("Mean Charges by Age Band")
        axes[0].set_ylabel("Mean Charges ($)")

        sns.boxplot(
            data=filtered_data, x="age_band", y="charges",
            color="#4C72B0", ax=axes[1],
            order=["18-25", "26-35", "36-45", "46-55", "56-65", "66-100"],
        )
        axes[1].set_title("Charge Distribution by Age Band")
        axes[1].set_xlabel("Age Band")
        axes[1].set_ylabel("Charges ($)")

        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # By BMI Category
        st.subheader("Charges by BMI Category")
        bmi_stats = filtered_data.groupby("bmi_category", observed=True)["charges"]
        bmi_stats = bmi_stats.agg(["count", "mean", "median"])
        bmi_stats = bmi_stats.round(2)
        bmi_stats = bmi_stats.reset_index()
        bmi_stats.columns = ["BMI Category", "Count", "Mean ($)", "Median ($)"]
        st.dataframe(bmi_stats, hide_index=True, width="stretch")

        # By Sex
        st.subheader("Charges by Sex")
        sex_stats = filtered_data.groupby("sex")["charges"]
        sex_stats = sex_stats.agg(["count", "mean", "median"])
        sex_stats = sex_stats.round(2)
        sex_stats = sex_stats.reset_index()
        sex_stats.columns = ["Sex", "Count", "Mean ($)", "Median ($)"]
        st.dataframe(sex_stats, hide_index=True, width="stretch")

        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_data = filtered_data[["age", "bmi", "children", "charges"]].copy()
        numeric_data["smoker_flag"] = (filtered_data["smoker"] == "yes").astype(int)

        figure, chart = plt.subplots(figsize=(7, 5))
        correlations = numeric_data.corr()
        sns.heatmap(
            correlations, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=chart,
        )
        chart.set_title("Pearson Correlation Matrix")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        st.caption(
            "Correlation shows linear association strength. "
            "A high correlation does not imply causation."
        )


# TAB 3 — Data Quality
with quality_tab:
    st.header("Data Quality Report")
    st.caption("This report uses the full dataset, regardless of sidebar filters.")

    st.subheader("Column Information")
    column_details = []
    for column in data.columns:
        column_details.append({
            "Column": column,
            "Non-Null Count": data[column].notna().sum(),
            "Null Count": data[column].isna().sum(),
            "Dtype": str(data[column].dtype),
            "Unique Values": data[column].nunique(),
        })
    column_information = pd.DataFrame(column_details)
    st.dataframe(column_information, hide_index=True, width="stretch")

    st.subheader("Missing Values")
    missing = data.isna().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in the dataset.")
    else:
        st.dataframe(
            missing[missing > 0].reset_index()
            .rename(columns={"index": "Column", 0: "Missing Count"}),
            hide_index=True,
        )

    st.subheader("Duplicate Rows")
    duplicate_count = data.duplicated().sum()
    if duplicate_count == 0:
        st.success("✅ No exact duplicate rows found.")
    else:
        st.warning(
            f"⚠️ {duplicate_count} duplicate row(s) detected. "
            "Investigate whether they are valid records or data errors."
        )
        st.dataframe(
            data[data.duplicated(keep=False)].sort_values(
                list(data.columns)
            ),
            hide_index=True,
        )

    st.subheader("Numeric Column Statistics")
    st.dataframe(
        data.describe().T.round(2), width="stretch",
    )

    st.subheader("Value Counts — Categorical Columns")
    category_columns = ["sex", "smoker", "region"]
    category_containers = st.columns(len(category_columns))
    for i, column_name in enumerate(category_columns):
        with category_containers[i]:
            st.markdown(f"**{column_name}**")
            value_counts = data[column_name].value_counts().reset_index()
            value_counts.columns = [column_name, "Count"]
            st.dataframe(value_counts, hide_index=True, width="stretch")

    st.subheader("Potential Outliers (charges > 99th percentile)")
    outlier_threshold = data["charges"].quantile(0.99)
    outliers = data[data["charges"] > outlier_threshold]
    st.write(
        f"Threshold (99th percentile): **${outlier_threshold:,.2f}** — "
        f"{len(outliers)} record(s) above this value."
    )
    if not outliers.empty:
        st.dataframe(outliers, hide_index=True, width="stretch")


# TAB 4 — Prediction & Evaluation
with prediction_tab:
    st.header("Predict Insurance Charges")
    st.caption("Models use the full dataset, regardless of sidebar filters. This is a learning estimate, not an insurance quote.")

    # Prediction form
    st.subheader("Enter details")
    input_column_1, input_column_2, input_column_3 = st.columns(3)

    with input_column_1:
        input_age = st.slider("Age", 18, 100, 30)
        input_sex = st.selectbox("Sex", ["male", "female"])

    with input_column_2:
        input_bmi = st.slider("BMI", 10.0, 55.0, 25.0, step=0.5)
        input_children = st.slider("Children", 0, 5, 0)

    with input_column_3:
        input_smoker = st.selectbox("Smoker?", ["no", "yes"])
        input_region = st.selectbox(
            "Region", ["northeast", "northwest", "southeast", "southwest"]
        )

    # Train a fresh model using ALL features
    st.markdown("---")
    show_evaluation = st.checkbox("📈 Train models and show evaluation", value=False)

    if show_evaluation:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Features are the six inputs. The target is the charge to predict.
        feature_columns = ["age", "sex", "bmi", "children", "smoker", "region"]
        features = data[feature_columns]
        target = data["charges"]

        # Keep 20% of rows aside to check predictions on unseen records.
        # A fixed random_state gives the same split each time.
        train_features, test_features, train_charges, test_charges = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Models need numbers. get_dummies creates 0/1 columns for text values.
        # For example, region_southeast is 1 for southeast and 0 otherwise.
        category_columns = ["sex", "smoker", "region"]
        train_features = train_features.copy()
        test_features = test_features.copy()
        training_categories = {}
        for column in category_columns:
            categories = sorted(train_features[column].unique())
            training_categories[column] = categories
            train_features[column] = pd.Categorical(train_features[column], categories=categories)
            test_features[column] = pd.Categorical(test_features[column], categories=categories)

        train_features = pd.get_dummies(train_features, drop_first=True, dtype=int)
        encoded_test_features = pd.get_dummies(test_features, drop_first=True, dtype=int)
        encoded_test_features = encoded_test_features.reindex(
            columns=train_features.columns, fill_value=0
        )

        # Models
        models = {
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
            ),
        }

        results = []
        best_model = None
        best_r2 = float("-inf")  # Start below any possible R² score.

        # fit() learns from training rows; predict() estimates held-out charges.
        for name, model in models.items():
            model.fit(train_features, train_charges)
            predictions = model.predict(encoded_test_features)
            mae = mean_absolute_error(test_charges, predictions)
            rmse = np.sqrt(mean_squared_error(test_charges, predictions))
            r2 = r2_score(test_charges, predictions)
            results.append({
                "Model": name,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "R²": round(r2, 4),
            })
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = name
                best_predictions = predictions

        st.subheader("Model Comparison (80/20 split, random_state=42)")
        model_results = pd.DataFrame(results)
        st.dataframe(model_results, hide_index=True, width="stretch")

        st.info(
            f"**Best model:** {best_model_name} — R² = {best_r2:.4f}\n\n"
            "R² represents the proportion of variance in charges explained by "
            "the model. MAE is the average absolute prediction error."
        )

        st.caption("The higher R² selects the model on this split; these scores are not an independent final evaluation.")

        # Actual vs Predicted
        st.subheader("Actual vs Predicted (Test Set)")
        figure, chart = plt.subplots(figsize=(7, 5))
        chart.scatter(test_charges, best_predictions, alpha=0.5, s=20, color="#4C72B0")
        line_min = min(test_charges.min(), best_predictions.min())
        line_max = max(test_charges.max(), best_predictions.max())
        chart.plot(
            [line_min, line_max], [line_min, line_max],
            "r--", linewidth=1.5, label="Perfect prediction",
        )
        chart.set_xlabel("Actual Charges ($)")
        chart.set_ylabel("Predicted Charges ($)")
        chart.set_title(f"{best_model_name} — Actual vs Predicted")
        chart.legend()
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # Residual analysis
        st.subheader("Residual Distribution")
        residuals = test_charges - best_predictions
        figure, chart = plt.subplots(figsize=(7, 4))
        chart.hist(residuals, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
        chart.axvline(0, color="red", linestyle="--", linewidth=1.5)
        chart.set_xlabel("Residual (Actual − Predicted) ($)")
        chart.set_ylabel("Frequency")
        chart.set_title("Residual Distribution")
        plt.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        # Error by segment
        st.subheader("Mean Absolute Error by Segment")
        test_results = test_features.copy()
        test_results["actual"] = test_charges.values
        test_results["predicted"] = best_predictions
        test_results["abs_error"] = np.abs(test_results["actual"] - test_results["predicted"])
        test_results["smoker_label"] = test_results["smoker"]

        segment_errors = test_results.groupby("smoker_label", observed=True)["abs_error"]
        segment_errors = segment_errors.agg(["count", "mean"])
        segment_errors = segment_errors.round(2)
        segment_errors = segment_errors.reset_index()
        segment_errors.columns = ["Smoker", "Test Records", "MAE ($)"]
        st.dataframe(segment_errors, hide_index=True, width="stretch")

        # Feature importance (Gradient Boosting)
        if hasattr(best_model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                "Feature": train_features.columns,
                "Importance": best_model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            figure, chart = plt.subplots(figsize=(7, 4))
            chart.barh(importance["Feature"], importance["Importance"], color="#4C72B0")
            chart.set_xlabel("Importance")
            chart.set_title("Feature Importance (Gradient Boosting)")
            plt.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

        # Prediction using the best trained model
        st.markdown("---")
        st.subheader("Predict with Trained Model")

        input_row = pd.DataFrame({
            "age": [input_age],
            "sex": [input_sex],
            "bmi": [input_bmi],
            "children": [input_children],
            "smoker": [input_smoker],
            "region": [input_region],
        })

        # Use the training categories and column order for the new record too.
        for column in category_columns:
            input_row[column] = pd.Categorical(
                input_row[column], categories=training_categories[column]
            )
        input_row = pd.get_dummies(input_row, drop_first=True, dtype=int)
        input_row = input_row.reindex(columns=train_features.columns, fill_value=0)

        if st.button("🔮 Predict Charges"):
            predicted_charge = best_model.predict(input_row)[0]
            st.success(f"### Estimated Charges: **${predicted_charge:,.2f}**")
            st.caption(f"Model used: {best_model_name}")

    else:
        st.caption(
            "Check the box above to train models and see evaluation metrics. "
            "The prediction will use the best-performing model."
        )

# Footer
st.markdown("---")
st.caption(
    "Dataset: [Medical Cost Personal Datasets]"
    "(https://www.kaggle.com/datasets/mirichoi0218/insurance) — "
    "1,338 records · 6 input features · 1 target (charges)"
)
