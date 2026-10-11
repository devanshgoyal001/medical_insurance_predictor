# Medical Insurance Charges: a beginner's project

This project explores a CSV file and estimates medical charges using machine learning. You can inspect the same data with pandas, SQL, and an interactive Streamlit dashboard.

## Start here if you have little time

Open `analysis.py`. It is a short script that runs from top to bottom:

**Load CSV → Explore rows → Check data quality → Summarize charges → Compare groups → Describe a finding**

With the virtual environment activated, run:

```bash
python analysis.py
```

Focus on just five pandas operations first: `read_csv`, `head`, `isna`, `mean`, and `groupby`.
The script prints its results and keeps the original CSV unchanged. It reports duplicates so you can review them before deciding whether to remove any records.

## Files and suggested reading order

1. `insurance.csv`: the original table, with 1,338 rows and seven columns.
2. `analysis.py`: start here for a short, six-step analysis pipeline.
3. `sql_analysis.py`: ten SQL reports using a temporary SQLite database.
4. `app.py`: the interactive dashboard, followed by model training and prediction.
5. `requirements.txt`: the Python packages needed to run the project.

The existing `.pkl` files are older saved models. The current scripts do not load them; the dashboard trains its own models. `.venv` contains installed packages, not project code.

## Run the project

Open a terminal in this project folder. Create a virtual environment once:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages, then start the dashboard:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To print the analysis reports instead:

```bash
python analysis.py
python sql_analysis.py
```

Press Ctrl+C in the terminal to stop the dashboard.

## Understand the data

| Column | Meaning |
| --- | --- |
| age | Age in years |
| sex | Recorded sex category |
| bmi | Body mass index |
| children | Number of children/dependents recorded |
| smoker | Smoking status: yes or no |
| region | Recorded region |
| charges | Recorded medical charges in dollars; the value we predict |

There are **six input features and one target**, not seven input features.

Source: [Medical Cost Personal Datasets on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## Understand the Python code

A pandas `DataFrame` is a table. In these scripts it is usually named `data`.

```python
# Select one column.
charges = data["charges"]

# Select only the rows where smoker is yes.
smoker_rows = data[data["smoker"] == "yes"]

# Group the records, then calculate an average for each group.
smoking_groups = data.groupby("smoker")["charges"]
average_charges = smoking_groups.mean()
```

Common operations used throughout the project:

| Code | Meaning |
| --- | --- |
| `pd.read_csv(path)` | Load a CSV as a table |
| `data.head()` | Show the first five rows |
| `data.isna().sum()` | Count missing values in each column |
| `data.duplicated().sum()` | Count repeated rows after the first occurrence |
| `.agg(["count", "mean", "median"])` | Calculate several summaries |
| `.reset_index()` | Turn group labels into ordinary columns |
| `.round(2)` | Round numbers to two decimal places |
| `.copy()` | Make a separate table to edit |
| `pd.cut(...)` | Put numbers into named ranges |
| `f"${amount:,.2f}"` | Display a dollar amount with commas and two decimals |

The simple `analysis.py` script runs from top to bottom. In `sql_analysis.py`, a function groups steps under a name; `main()` holds the report, and `if __name__ == "__main__":` runs it when you start that file directly.

## Understand the dashboard

Read `app.py` from top to bottom:

1. Set the page title and load the CSV. `@st.cache_data` lets Streamlit reuse loaded data.
2. Add age and BMI groups for comparisons.
3. Read sidebar choices and filter rows one condition at a time.
4. Display the four tabs.
5. Train and compare models only when the evaluation checkbox is selected.

| Tab | What you can do |
| --- | --- |
| Overview | See charge summaries, distributions, scatter plots, and percentiles |
| Group Comparisons | Compare smoking, region, age, BMI, and sex groups; inspect correlations |
| Data Quality | Inspect missing values, duplicates, types, counts, and potential outliers |
| Prediction & Evaluation | Compare models, inspect errors and feature importance, estimate a new charge |

Sidebar filters affect Overview and Group Comparisons. Data Quality and model training use the full dataset. Empty filter results show a message.

For charts, `figure` is the full image and `chart` is the plotting area. `st.pyplot(figure)` displays it, and `plt.close(figure)` releases its memory.

Streamlit reruns the script when a widget changes. With evaluation enabled, the small models train again on each rerun. This keeps the learning example straightforward.

## Understand the prediction steps

1. Select the six input columns as `features` and charges as `target`.
2. Split rows into 80% training data and 20% held-out data. `random_state=42` makes the split repeatable.
3. Convert text categories into 0/1 columns using `pd.get_dummies`. One category is omitted as the reference group. Use the training categories and the same column order for test rows and new inputs.
4. Train Linear Regression and Gradient Boosting using `fit()`.
5. Use `predict()` on the held-out rows and compare the estimates with actual charges.
6. Select the model with the higher R² on this split and use it for the entered details.

Linear Regression learns a weighted sum of inputs. Gradient Boosting builds small decision trees in sequence to improve earlier predictions. The code uses 200 trees, a maximum depth of 4, and a learning rate of 0.1.

| Measure | Meaning |
| --- | --- |
| MAE | Average absolute prediction error in dollars; lower is better |
| RMSE | Error in dollars that gives more weight to large mistakes; lower is better |
| R² | Score relative to predicting the test-set mean; 1 is perfect, 0 matches that baseline, and negative is worse |
| Residual | Actual charge minus predicted charge |

The dashboard also compares errors between smoking groups. When Gradient Boosting wins, it displays feature importance for the encoded input columns.

Use the displayed results when discussing performance. R² is not classification accuracy. Because this same held-out split selects the model, its score is a comparison score rather than an independent final evaluation. A later improvement would use validation data or cross-validation to select the model and a separate test set for final evaluation.

This small dataset cannot establish causation or guarantee an individual's future costs. Inputs beyond the ages and BMI values in the data may give unreliable estimates. The output is a learning estimate, not an insurance quote.

## Understand the SQL script

The script copies the CSV into SQLite's `:memory:` database. This exists only while the script runs. Each report prints its SQL and result.

- `SELECT` chooses columns or calculations.
- `GROUP BY` groups records before calculating `COUNT`, `AVG`, `MIN`, `MAX`, or `SUM`.
- `CASE WHEN` creates categories or conditional calculations.
- `HAVING` filters grouped results, such as repeated rows.
- A subquery calculates a total used in a percentage.
- `WITH` names an intermediate result so the next query can use it.
- `OVER (PARTITION BY region)` calculates within each region while keeping individual rows.
- `DENSE_RANK` ranks distinct charge values within each region. Ties can produce more than three records for the top three ranks.

Start with queries 1–4, then study window functions in queries 5–6.

## Practice before adding it to your resume

Try changing a chart color, adding a median to a SQL report, and explaining why the model needs held-out rows. Follow a single prediction from the input widgets through encoding to `predict()`.

Once you can explain and demonstrate the work, adapt this resume bullet:

> Built a Streamlit dashboard to explore 1,338 medical insurance records using pandas and SQLite; compared Linear Regression and Gradient Boosting using MAE, RMSE, and R², and added an interactive charge estimator.

Include measured scores only after running the app and understanding what they mean.
