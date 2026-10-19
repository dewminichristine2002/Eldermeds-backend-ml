"""
Exploratory Data Analysis (EDA)
FDA Adverse Event Reporting System (FAERS)

This script performs descriptive exploratory analysis only.
It does NOT train, tune, or modify the ML models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "fda_adverse_events_2015_2026_CLEAN.csv"

OUTPUT_DIR = ROOT / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FDA ADVERSE EVENT DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nLoading dataset from:\n{DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SUMMARY")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\nColumn names:")
for column in df.columns:
    print(f" - {column}")


summary = pd.DataFrame(
    {
        "Metric": [
            "Number of rows",
            "Number of columns",
            "Duplicate rows",
        ],
        "Value": [
            len(df),
            len(df.columns),
            df.duplicated().sum(),
        ],
    }
)

summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)


# ============================================================
# 2. TARGET VARIABLE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. SERIOUS / NON-SERIOUS CLASS DISTRIBUTION")
print("=" * 70)

if "serious" in df.columns:
    serious_clean = df["serious"].fillna("Missing").astype(str).str.strip()

    class_counts = serious_clean.value_counts(dropna=False)
    class_percentages = serious_clean.value_counts(normalize=True, dropna=False) * 100

    class_table = pd.DataFrame(
        {
            "Count": class_counts,
            "Percentage": class_percentages,
        }
    )

    print(class_table)

    class_table.to_csv(OUTPUT_DIR / "class_distribution.csv")

    plt.figure(figsize=(7, 5))

    class_counts.plot(kind="bar")

    plt.title("Distribution of Serious and Non-Serious Reports")
    plt.xlabel("Serious Classification")
    plt.ylabel("Number of Reports")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = df.isnull().sum() / len(df) * 100

missing_table = pd.DataFrame(
    {
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage,
    }
)

missing_table = missing_table.sort_values("Missing_Percentage", ascending=False)

print(missing_table)

missing_table.to_csv(OUTPUT_DIR / "missing_values.csv")


# Plot only columns containing missing values

missing_plot = missing_table[missing_table["Missing_Count"] > 0]

if not missing_plot.empty:
    plt.figure(figsize=(11, 7))

    missing_plot["Missing_Percentage"].plot(kind="bar")

    plt.title("Missing Values by Variable")
    plt.xlabel("Variable")
    plt.ylabel("Missing Values (%)")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_values.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 4. PATIENT AGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. PATIENT AGE DISTRIBUTION")
print("=" * 70)

if "patient_age_years" in df.columns:
    age = pd.to_numeric(df["patient_age_years"], errors="coerce")

    # Same plausible age boundary used by model pipeline
    age = age.where((age >= 0) & (age <= 100))

    print(age.describe())

    age.describe().to_csv(OUTPUT_DIR / "age_statistics.csv")

    plt.figure(figsize=(8, 5))

    plt.hist(age.dropna(), bins=20, edgecolor="black")

    plt.title("Distribution of Patient Age")
    plt.xlabel("Age (Years)")
    plt.ylabel("Number of Reports")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "age_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 5. PATIENT SEX DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. PATIENT SEX DISTRIBUTION")
print("=" * 70)

if "patient_sex" in df.columns:
    sex = (
        df["patient_sex"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace(
            {
                "M": "Male",
                "F": "Female",
                "UNK": "Unknown",
            }
        )
    )

    sex_counts = sex.value_counts()

    print(sex_counts)

    sex_counts.to_csv(
        OUTPUT_DIR / "sex_distribution.csv",
        header=["Count"],
    )

    plt.figure(figsize=(7, 5))

    sex_counts.plot(kind="bar")

    plt.title("Patient Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Number of Reports")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sex_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 6. NUMBER OF DRUGS PER REPORT
# ============================================================

print("\n" + "=" * 70)
print("6. NUMBER OF DRUGS PER REPORT")
print("=" * 70)

if "num_drugs" in df.columns:
    num_drugs = pd.to_numeric(df["num_drugs"], errors="coerce")

    print(num_drugs.describe())

    num_drugs.describe().to_csv(OUTPUT_DIR / "num_drugs_statistics.csv")

    # Restrict plot to 99th percentile for readability.
    drug_limit = num_drugs.quantile(0.99)

    plot_drugs = num_drugs[num_drugs <= drug_limit].dropna()

    plt.figure(figsize=(8, 5))

    plt.hist(plot_drugs, bins=20, edgecolor="black")

    plt.title("Distribution of Number of Drugs per Report")
    plt.xlabel("Number of Drugs")
    plt.ylabel("Number of Reports")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "num_drugs_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 7. NUMBER OF REACTIONS
# ============================================================

print("\n" + "=" * 70)
print("7. NUMBER OF REACTIONS PER REPORT")
print("=" * 70)

if "num_reactions" in df.columns:
    num_reactions = pd.to_numeric(df["num_reactions"], errors="coerce")

    print(num_reactions.describe())

    num_reactions.describe().to_csv(OUTPUT_DIR / "num_reactions_statistics.csv")

    reaction_limit = num_reactions.quantile(0.99)

    plot_reactions = num_reactions[num_reactions <= reaction_limit].dropna()

    plt.figure(figsize=(8, 5))

    plt.hist(plot_reactions, bins=20, edgecolor="black")

    plt.title("Distribution of Number of Reactions per Report")
    plt.xlabel("Number of Reactions")
    plt.ylabel("Number of Reports")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "num_reactions_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 8. TOP PRIMARY REACTIONS
# ============================================================

print("\n" + "=" * 70)
print("8. TOP PRIMARY REACTIONS")
print("=" * 70)

if "primary_reaction" in df.columns:
    reaction_series = df["primary_reaction"].fillna("Unknown").astype(str).str.strip()

    top_reactions = reaction_series.value_counts().head(15)

    print(top_reactions)

    top_reactions.to_csv(
        OUTPUT_DIR / "top_primary_reactions.csv",
        header=["Count"],
    )

    plt.figure(figsize=(10, 7))

    top_reactions.sort_values().plot(kind="barh")

    plt.title("Top 15 Primary Adverse Reactions")
    plt.xlabel("Number of Reports")
    plt.ylabel("Primary Reaction")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_primary_reactions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 9. TOP SUSPECT DRUGS
# ============================================================

print("\n" + "=" * 70)
print("9. TOP SUSPECT DRUGS")
print("=" * 70)

if "suspect_drug" in df.columns:
    drug_series = df["suspect_drug"].fillna("Unknown").astype(str).str.strip()

    top_drugs = drug_series.value_counts().head(15)

    print(top_drugs)

    top_drugs.to_csv(
        OUTPUT_DIR / "top_suspect_drugs.csv",
        header=["Count"],
    )

    plt.figure(figsize=(10, 7))

    top_drugs.sort_values().plot(kind="barh")

    plt.title("Top 15 Suspect Drugs")
    plt.xlabel("Number of Reports")
    plt.ylabel("Suspect Drug")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_suspect_drugs.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 10. SERIOUS EVENT RATE BY SEX
# ============================================================

print("\n" + "=" * 70)
print("10. SERIOUS EVENT RATE BY SEX")
print("=" * 70)

if "patient_sex" in df.columns and "serious" in df.columns:
    temp = df[["patient_sex", "serious"]].copy()

    temp["patient_sex"] = (
        temp["patient_sex"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace(
            {
                "M": "Male",
                "F": "Female",
                "UNK": "Unknown",
            }
        )
    )

    temp["serious_binary"] = temp["serious"].astype(str).str.strip().map(
        {
            "Yes": 1,
            "No": 0,
            "1": 1,
            "0": 0,
        }
    )

    serious_by_sex = temp.dropna(subset=["serious_binary"]).groupby("patient_sex")["serious_binary"].agg(["count", "mean"])

    serious_by_sex["Serious_Percentage"] = serious_by_sex["mean"] * 100

    print(serious_by_sex)

    serious_by_sex.to_csv(OUTPUT_DIR / "serious_rate_by_sex.csv")

    plt.figure(figsize=(7, 5))

    serious_by_sex["Serious_Percentage"].plot(kind="bar")

    plt.title("Serious Adverse Event Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Serious Reports (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "serious_rate_by_sex.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 11. SERIOUS EVENT RATE BY AGE GROUP
# ============================================================

print("\n" + "=" * 70)
print("11. SERIOUS EVENT RATE BY AGE GROUP")
print("=" * 70)

if "patient_age_years" in df.columns and "serious" in df.columns:
    temp = df[["patient_age_years", "serious"]].copy()

    temp["patient_age_years"] = pd.to_numeric(temp["patient_age_years"], errors="coerce")

    temp.loc[(temp["patient_age_years"] < 0) | (temp["patient_age_years"] > 100), "patient_age_years"] = np.nan

    temp["age_group"] = pd.cut(
        temp["patient_age_years"],
        bins=[0, 17, 39, 59, 64, 74, 84, 100],
        labels=[
            "0-17",
            "18-39",
            "40-59",
            "60-64",
            "65-74",
            "75-84",
            "85-100",
        ],
        include_lowest=True,
    )

    temp["serious_binary"] = temp["serious"].astype(str).str.strip().map(
        {
            "Yes": 1,
            "No": 0,
            "1": 1,
            "0": 0,
        }
    )

    serious_by_age = temp.dropna(subset=["age_group", "serious_binary"]).groupby(
        "age_group",
        observed=False,
    )["serious_binary"].agg(["count", "mean"])

    serious_by_age["Serious_Percentage"] = serious_by_age["mean"] * 100

    print(serious_by_age)

    serious_by_age.to_csv(OUTPUT_DIR / "serious_rate_by_age_group.csv")

    plt.figure(figsize=(9, 5))

    serious_by_age["Serious_Percentage"].plot(kind="bar")

    plt.title("Serious Adverse Event Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Serious Reports (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "serious_rate_by_age_group.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 12. SERIOUS RATE BY NUMBER OF DRUGS
# ============================================================

print("\n" + "=" * 70)
print("12. SERIOUS EVENT RATE BY NUMBER OF DRUGS")
print("=" * 70)

if "num_drugs" in df.columns and "serious" in df.columns:
    temp = df[["num_drugs", "serious"]].copy()

    temp["num_drugs"] = pd.to_numeric(temp["num_drugs"], errors="coerce")

    temp["drug_group"] = pd.cut(
        temp["num_drugs"],
        bins=[-np.inf, 1, 4, 9, np.inf],
        labels=[
            "1 drug",
            "2-4 drugs",
            "5-9 drugs",
            "10+ drugs",
        ],
    )

    temp["serious_binary"] = temp["serious"].astype(str).str.strip().map(
        {
            "Yes": 1,
            "No": 0,
            "1": 1,
            "0": 0,
        }
    )

    serious_by_drugs = temp.dropna(subset=["drug_group", "serious_binary"]).groupby(
        "drug_group",
        observed=False,
    )["serious_binary"].agg(["count", "mean"])

    serious_by_drugs["Serious_Percentage"] = serious_by_drugs["mean"] * 100

    print(serious_by_drugs)

    serious_by_drugs.to_csv(OUTPUT_DIR / "serious_rate_by_num_drugs.csv")

    plt.figure(figsize=(8, 5))

    serious_by_drugs["Serious_Percentage"].plot(kind="bar")

    plt.title("Serious Adverse Event Rate by Number of Drugs")
    plt.xlabel("Number of Drugs")
    plt.ylabel("Serious Reports (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "serious_rate_by_num_drugs.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 13. NUMERIC CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("13. NUMERIC FEATURE CORRELATION")
print("=" * 70)

numeric_columns = [
    "year",
    "month",
    "num_reactions",
    "num_drugs",
    "patient_age_years",
    "report_age_days",
]

available_numeric = [col for col in numeric_columns if col in df.columns]

numeric_df = df[available_numeric].apply(pd.to_numeric, errors="coerce")

correlation_matrix = numeric_df.corr()

print(correlation_matrix)

correlation_matrix.to_csv(OUTPUT_DIR / "numeric_correlations.csv")


# ============================================================
# 14. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"\nAll EDA tables and figures were saved to:\n{OUTPUT_DIR}")

print("\nGenerated evidence includes:")
print(" - Dataset summary")
print(" - Class distribution")
print(" - Missing-value analysis")
print(" - Patient age distribution")
print(" - Patient sex distribution")
print(" - Number of drugs distribution")
print(" - Number of reactions distribution")
print(" - Top primary reactions")
print(" - Top suspect drugs")
print(" - Serious-event rate by sex")
print(" - Serious-event rate by age")
print(" - Serious-event rate by number of drugs")
print(" - Numeric correlation matrix")
