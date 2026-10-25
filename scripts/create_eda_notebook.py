import json

def create_eda_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1: Dataset Inspection & Exploratory Data Analysis (EDA)\n",
                    "## Project: Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours\n",
                    "**Dataset:** PhysioNet / Computing in Cardiology Challenge 2012\n",
                    "\n",
                    "### Objectives of Phase 1:\n",
                    "1. Inspect official challenge files (`set-a`, `set-b`, `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt`).\n",
                    "2. Verify patient record alignment, ID format, timestamp structure, and recording duration (48 hours).\n",
                    "3. Profile patient cohort demographics, outcome distribution (in-hospital mortality), and class imbalance.\n",
                    "4. Assess measurement frequency, patient-level and temporal missingness across all 37 clinical parameters.\n",
                    "5. Identify duplicate measurements and physiological anomaly ranges (sensor disconnections, negative values, extreme outliers).\n",
                    "6. Formulate a rigorous, leak-free preprocessing and imputation strategy for Phase 2."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import glob\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "print('Libraries imported successfully.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Outcomes Inspection & Mortality Distribution\n",
                    "The challenge provides outcomes for Set A, Set B, and Set C. Each contains 4,000 patient records."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_a = pd.read_csv('../Dataset/Outcomes-a.txt')\n",
                    "df_b = pd.read_csv('../Dataset/Outcomes-b.txt')\n",
                    "df_c = pd.read_csv('../Dataset/Outcomes-c.txt')\n",
                    "\n",
                    "df_a['Set'] = 'Set A (Train)'\n",
                    "df_b['Set'] = 'Set B (Test/Leaderboard)'\n",
                    "df_c['Set'] = 'Set C (Held-out)'\n",
                    "\n",
                    "combined_outcomes = pd.concat([df_a, df_b, df_c], ignore_index=True)\n",
                    "\n",
                    "print('Outcomes Summary by Dataset Set:')\n",
                    "summary_tbl = combined_outcomes.groupby('Set').agg(\n",
                    "    Total_Patients=('RecordID', 'count'),\n",
                    "    Survivors=('In-hospital_death', lambda x: (x == 0).sum()),\n",
                    "    Deaths=('In-hospital_death', lambda x: (x == 1).sum()),\n",
                    "    Mortality_Rate_Pct=('In-hospital_death', lambda x: x.mean() * 100)\n",
                    ").reset_index()\n",
                    "display(summary_tbl)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Temporal Structure & Measurement Parsing\n",
                    "Each patient's record is stored as a text file where:\n",
                    "- `00:00` contains static variables (`RecordID`, `Age`, `Gender`, `Height`, `ICUType`, `Weight`).\n",
                    "- Dynamic physiological measurements follow with format `Time (HH:MM), Parameter, Value`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Preview a sample patient file\n",
                    "sample_path = '../Dataset/set-a/132539.txt'\n",
                    "sample_df = pd.read_csv(sample_path)\n",
                    "print('First 15 lines of patient 132539:')\n",
                    "display(sample_df.head(15))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Displaying Generated Publication EDA Figures\n",
                    "The generated high-resolution visualizations illustrate the distribution of outcomes, missingness, observation counts, and temporal trajectories."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "\n",
                    "fig_paths = [\n",
                    "    '../reports/figures/01_mortality_distribution.png',\n",
                    "    '../reports/figures/02_variable_coverage.png',\n",
                    "    '../reports/figures/03_observation_frequency.png',\n",
                    "    '../reports/figures/04_vitals_by_outcome.png',\n",
                    "    '../reports/figures/05_temporal_trajectories_48h.png',\n",
                    "    '../reports/figures/06_correlation_heatmap.png'\n",
                    "]\n",
                    "\n",
                    "for p in fig_paths:\n",
                    "    if os.path.exists(p):\n",
                    "        print(f'Displaying: {p}')\n",
                    "        display(Image(filename=p))\n",
                    "    else:\n",
                    "        print(f'Figure not found: {p}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.13.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open('D:/ICU_Prediction/notebooks/01_EDA.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/01_EDA.ipynb successfully.")

if __name__ == '__main__':
    create_eda_notebook()
