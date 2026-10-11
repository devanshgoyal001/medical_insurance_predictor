"""
SQL Analysis of Medical Insurance Data
========================================
Loads insurance.csv into an in-memory SQLite database and runs
analytical queries that demonstrate SQL skills.

Run:  python sql_analysis.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

def run_query(connection, title, sql):
    """Execute a query, print the title and result."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"SQL:\n{sql.strip()}\n")
    result = pd.read_sql_query(sql, connection)
    print(result.to_string(index=False))
    print()



def main():
    """Load the CSV, run the ten reports, then close the temporary database."""
    base_directory = Path(__file__).resolve().parent
    data = pd.read_csv(base_directory / "insurance.csv")

    # Load into SQLite
    connection = sqlite3.connect(":memory:")
    data.to_sql("insurance", connection, index=False, if_exists="replace")

    # Query 1: GROUP BY puts rows from the same region together.
    run_query(
        connection,
        "1. Average Charges by Region (sorted descending)",
        """
        SELECT
            region,
            COUNT(*)                    AS record_count,
            ROUND(AVG(charges), 2)      AS avg_charges,
            ROUND(MIN(charges), 2)      AS min_charges,
            ROUND(MAX(charges), 2)      AS max_charges
        FROM insurance
        GROUP BY region
        ORDER BY avg_charges DESC;
        """
    )

    # Query 2: Average Charges by Age Band (CASE expression)
    run_query(
        connection,
        "2. Average Charges by Age Band",
        """
        SELECT
            CASE
                WHEN age BETWEEN 18 AND 25 THEN '18-25'
                WHEN age BETWEEN 26 AND 35 THEN '26-35'
                WHEN age BETWEEN 36 AND 45 THEN '36-45'
                WHEN age BETWEEN 46 AND 55 THEN '46-55'
                WHEN age BETWEEN 56 AND 65 THEN '56-65'
                ELSE '66-100'
            END                         AS age_band,
            COUNT(*)                    AS record_count,
            ROUND(AVG(charges), 2)      AS avg_charges,
            ROUND(AVG(CASE WHEN smoker = 'yes' THEN charges END), 2)
                                        AS avg_charges_smoker,
            ROUND(AVG(CASE WHEN smoker = 'no'  THEN charges END), 2)
                                        AS avg_charges_nonsmoker
        FROM insurance
        GROUP BY age_band
        ORDER BY age_band;
        """
    )

    # Query 3: Smoking Group Comparison with Conditional Aggregation
    run_query(
        connection,
        "3. Smoker vs Non-Smoker Comparison",
        """
        SELECT
            smoker,
            COUNT(*)                    AS record_count,
            ROUND(AVG(charges), 2)      AS avg_charges,
            ROUND(AVG(bmi), 2)          AS avg_bmi,
            ROUND(AVG(age), 1)          AS avg_age,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM insurance), 1
            )                           AS pct_of_total
        FROM insurance
        GROUP BY smoker;
        """
    )

    # Query 4: Each Region's Percentage of Total Charges
    run_query(
        connection,
        "4. Each Region's Share of Total Charges",
        """
        SELECT
            region,
            COUNT(*)                        AS record_count,
            ROUND(SUM(charges), 2)          AS total_charges,
            ROUND(
                100.0 * SUM(charges) / (SELECT SUM(charges) FROM insurance), 2
            )                               AS pct_of_total_charges
        FROM insurance
        GROUP BY region
        ORDER BY pct_of_total_charges DESC;
        """
    )

    # DENSE_RANK gives equal charges the same rank, so ties can return more than 3 rows.
    # WITH gives the first query a name so the next query can filter its result.
    # Query 5: Top 3 Charges in Each Region (Window Function)
    run_query(
        connection,
        "5. Top 3 Highest Charges per Region (DENSE_RANK)",
        """
        WITH ranked AS (
            SELECT
                region,
                age,
                sex,
                smoker,
                bmi,
                charges,
                DENSE_RANK() OVER (
                    PARTITION BY region ORDER BY charges DESC
                ) AS charge_rank
            FROM insurance
        )
        SELECT *
        FROM ranked
        WHERE charge_rank <= 3
        ORDER BY region, charge_rank;
        """
    )

    # OVER (PARTITION BY region) calculates an average without combining the rows.
    # Query 6: Records Above Their Region's Average
    run_query(
        connection,
        "6. Count of Records Above Their Own Region's Average (Window Function)",
        """
        WITH region_avg AS (
            SELECT
                *,
                AVG(charges) OVER (PARTITION BY region) AS region_avg_charges
            FROM insurance
        )
        SELECT
            region,
            COUNT(*)                                            AS above_avg_count,
            ROUND(AVG(charges), 2)                              AS avg_charges_above,
            ROUND(MAX(region_avg_charges), 2)                   AS region_overall_avg
        FROM region_avg
        WHERE charges > region_avg_charges
        GROUP BY region
        ORDER BY above_avg_count DESC;
        """
    )

    # Query 7: Duplicate and Missing-Value Checks
    run_query(
        connection,
        "7. Exact Duplicate Row Check",
        """
        SELECT
            age, sex, bmi, children, smoker, region, charges,
            COUNT(*) AS occurrences
        FROM insurance
        GROUP BY age, sex, bmi, children, smoker, region, charges
        HAVING COUNT(*) > 1
        ORDER BY occurrences DESC
        LIMIT 10;
        """
    )

    run_query(
        connection,
        "8. NULL Value Check per Column",
        """
        SELECT
            SUM(CASE WHEN age      IS NULL THEN 1 ELSE 0 END) AS age_nulls,
            SUM(CASE WHEN sex      IS NULL THEN 1 ELSE 0 END) AS sex_nulls,
            SUM(CASE WHEN bmi      IS NULL THEN 1 ELSE 0 END) AS bmi_nulls,
            SUM(CASE WHEN children IS NULL THEN 1 ELSE 0 END) AS children_nulls,
            SUM(CASE WHEN smoker   IS NULL THEN 1 ELSE 0 END) AS smoker_nulls,
            SUM(CASE WHEN region   IS NULL THEN 1 ELSE 0 END) AS region_nulls,
            SUM(CASE WHEN charges  IS NULL THEN 1 ELSE 0 END) AS charges_nulls
        FROM insurance;
        """
    )

    # Query 9: BMI Category Analysis
    run_query(
        connection,
        "9. Charges by BMI Category",
        """
        SELECT
            CASE
                WHEN bmi < 18.5 THEN 'Underweight'
                WHEN bmi < 25   THEN 'Normal'
                WHEN bmi < 30   THEN 'Overweight'
                ELSE 'Obese'
            END                         AS bmi_category,
            COUNT(*)                    AS record_count,
            ROUND(AVG(charges), 2)      AS avg_charges,
            SUM(CASE WHEN smoker = 'yes' THEN 1 ELSE 0 END) AS smoker_count,
            ROUND(
                100.0 * SUM(CASE WHEN smoker = 'yes' THEN 1 ELSE 0 END) / COUNT(*), 1
            ) AS smoker_pct
        FROM insurance
        GROUP BY bmi_category
        ORDER BY avg_charges DESC;
        """
    )

    # Query 10: Count records in each charge range
    run_query(
        connection,
        "10. Charge Buckets — Distribution",
        """
        SELECT
            CASE
                WHEN charges < 5000  THEN '< $5K'
                WHEN charges < 10000 THEN '$5K-$10K'
                WHEN charges < 20000 THEN '$10K-$20K'
                WHEN charges < 30000 THEN '$20K-$30K'
                WHEN charges < 40000 THEN '$30K-$40K'
                ELSE '$40K+'
            END                         AS charge_bucket,
            COUNT(*)                    AS record_count,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM insurance), 1
            )                           AS pct_of_total
        FROM insurance
        GROUP BY charge_bucket
        ORDER BY MIN(charges);
        """
    )

    connection.close()
    print("\n✅ All SQL queries executed successfully.\n")


# Run the reports only when this file is started directly.
if __name__ == "__main__":
    main()
