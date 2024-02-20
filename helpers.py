import glob
import os
import ast  # To safely evaluate the string representation of the lists
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr


Path("data").mkdir(exist_ok=True)
Path("data/byfield").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42


def compute_average_citation_age_from_field_to_other_fields(
    field: str = "NLP",
):
    """
    Computes the average citation age from a specified field to other fields.

    Args:
        field (str, optional): The field for which to compute the citation age. Defaults to "NLP".

    Returns:
        None

    Example:
        ```python
        compute_citation_age_from_field_to_other_fields("NLP")
        ```"""
    # Read output/NLP_paper_to_stats.csv
    df = pd.read_csv(f"data/byfield/{field}_paper_to_stats.csv")

    # Drop na for outgoing_citation_ages and outgoing_fields
    df = df.dropna(subset=["outgoing_citation_ages", "outgoing_fields"])

    # Convert the string representations of lists to actual lists
    df["outgoing_citation_ages"] = df["outgoing_citation_ages"].apply(ast.literal_eval)
    df["outgoing_fields"] = df["outgoing_fields"].apply(ast.literal_eval)

    # Explode outgoing_citation_ages
    df = df.explode("outgoing_citation_ages")

    # Explode outgoing_fields twice
    df = df.explode("outgoing_fields")  # This explodes the outer list
    df = df.explode("outgoing_fields")  # This explodes the inner list

    # Compute the average citation age from NLP to other fields
    average_citation_age = df.groupby("outgoing_fields")[
        "outgoing_citation_ages"
    ].mean()

    # Export to output/NLP_average_citation_age_to_field.csv
    average_citation_age.to_csv(
        f"data/byfield/{field}_average_citation_age_to_field.csv"
    )

    print(average_citation_age.head())


def compute_citation_age_from_own_field_and_to_other_fields(
    fields: list = [
        "NLP",
        "ML",
        "Psychology",
        "Sociology",
        "Linguistics",
        "Mathematics",
        "Computer science",
    ],
):
    """
    Computes the average citation age from a field to itself and to other fields.

    Args:
        fields (list, optional): A list of fields for which to compute the citation age. Defaults to ["NLP", "ML", "Psychology", "Sociology", "Linguistics", "Mathematics"].

    Returns:
        None

    Example:
        ```python
        compute_citation_age_from_own_field_and_to_other_fields(["NLP", "ML"])
        ```
    """
    for field in fields:
        # Read output/NLP_paper_to_stats.csv
        df = pd.read_csv(f"data/byfield/{field}_paper_to_stats.csv")

        # Drop na for outgoing_citation_ages and outgoing_fields
        df = df.dropna(subset=["outgoing_citation_ages", "outgoing_fields"])

        # Convert the string representations of lists to actual lists
        df["outgoing_citation_ages"] = df["outgoing_citation_ages"].apply(
            ast.literal_eval
        )
        df["outgoing_fields"] = df["outgoing_fields"].apply(ast.literal_eval)

        # Explode outgoing_citation_ages
        df = df.explode("outgoing_citation_ages")

        # Explode outgoing_fields twice
        df = df.explode("outgoing_fields")  # This explodes the outer list
        df = df.explode("outgoing_fields")  # This explodes the inner list

        # Compute the average citation age from field to itself (where outgoing_fields == field)
        average_citation_age_self = (
            df[
                (
                    df["outgoing_fields"] == field
                    if field not in ["NLP", "ML"]
                    else df["outgoing_fields"] == "Computer science"
                )
            ]
            .groupby("year")["outgoing_citation_ages"]
            .mean()
            .rename("average_citation_age_to_self")
        )

        # Compute the average citation age from field to other fields per year (where outgoing_fields != field)
        average_citation_age_other = (
            df[df["outgoing_fields"] != field]
            .groupby("year")["outgoing_citation_ages"]
            .mean()
        ).rename("average_citation_age_to_other_fields")

        # Combine results into one DataFrame
        result = pd.concat(
            [average_citation_age_self, average_citation_age_other], axis=1
        )

        # Export to output/{field}_average_citation_age_to_self_and_others.csv
        result.reset_index().to_csv(
            f"data/byfield/{field}_average_citation_age_to_self_and_others.csv",
            index=False,
        )


def _sum_citations_in_epoch(citations_list, epoch_start, epoch_end):
    """
    Calculates the sum of the "cited_by_count" values for citations within a specified epoch.

    Args:
        citations_list (list): A list of dictionaries representing citations, each containing a "cited_by_count" and "year" key.
        epoch_start (int): The start year of the epoch (inclusive).
        epoch_end (int): The end year of the epoch (exclusive).

    Returns:
        int: The sum of the "cited_by_count" values for citations within the specified epoch.

    Example:
        ```python
        citations = [
            {"cited_by_count": 10, "year": 2010},
            {"cited_by_count": 5, "year": 2015},
            {"cited_by_count": 8, "year": 2020},
            {"cited_by_count": 3, "year": 2025},
        ]
        epoch_start = 2015
        epoch_end = 2025

        sum_citations = _sum_citations_in_epoch(citations, epoch_start, epoch_end)
        print(sum_citations)  # Output: 11
        ```"""
    return sum(
        d["cited_by_count"]
        for d in citations_list
        if epoch_start <= d["year"] < epoch_end
    )


def compute_rankings(field):
    # Load the data
    df = pd.read_csv(f"data/byfield/{field}_paper_to_stats.csv")[:10000]

    # Define time periods for analysis
    epochs = [(1980, 1990), (1990, 2000), (2000, 2010), (2010, 2015), (2015, 2020)]

    # Initialize a DataFrame to store the rankings with all unique paper_ids
    rankings_df = pd.DataFrame()

    # Iterate through each combination of publishing and citing epochs
    for i, publish_epoch in enumerate(epochs[:-1]):
        for cite_epoch in epochs[i + 1 :]:
            # Column name for the current combination
            column_name = (
                f"papers_from_{publish_epoch}_ranked_by_citations_from_{cite_epoch}"
            )

            # Filter papers published in the publishing epoch
            filtered_papers = df[
                df["year"].between(publish_epoch[0], publish_epoch[1])
            ].copy()

            # Initialize an empty list to store the citation counts for each paper
            citation_counts = []

            # Iterate through each row in the filtered_papers DataFrame
            for index, row in filtered_papers.iterrows():
                # Initialize the citation count for the current paper
                current_citation_count = 0
                
                # Convert the string representation of the list to an actual list
                counts_by_year = ast.literal_eval(row['counts_by_year'])

                # Iterate through each count dictionary in the counts_by_year list
                for count in counts_by_year:
                    # Check if the year of the citation is within the citing epoch
                    if cite_epoch[0] <= count['year'] < 2023:
                        # Add the cited_by_count to the current paper's citation count
                        current_citation_count += int(count['cited_by_count'])

                # Append the calculated citation count for the current paper to the list
                citation_counts.append(current_citation_count)

            # Add the calculated citation counts as a new column to the filtered_papers DataFrame
            filtered_papers[column_name + "_count"] = citation_counts

            print(filtered_papers[column_name + "_count"].isna().count())

            print(filtered_papers[column_name + "_count"].count())

            # Fill filtered papers in column name with 0s if NaN
            # filtered_papers[column_name + "_count"] = filtered_papers[
            #     column_name + "_count"
            # ].fillna(0)

            # Sort the papers by the citation counts
            filtered_papers = filtered_papers.sort_values(
                by=column_name + "_count", ascending=False
            )

            # Rename the paper_id column to the column_name
            filtered_papers = filtered_papers.rename(columns={"paper_id": column_name})

            # Add the paper_ids to the rankings DataFrame
            rankings_df = pd.concat(
                [rankings_df, filtered_papers[column_name].reset_index(drop=True)],
                axis=1,
            )

    # Save the rankings to a CSV file
    rankings_df.to_csv(f"output/{field}_rankings.csv", index=False)

    return rankings_df


def compute_ranking_correlations(field: str = "NLP", percentile=0.1):
    """Computes Spearman, Kendall Tau, and RBO ranking correlations for a given field and percentile.

    Args:
        field (str): The field for which rankings are computed. Defaults to "NLP".
        percentile (float): The percentile of papers to consider. Defaults to 0.5.

    Returns:
        dict: A dictionary containing three pandas DataFrames for Spearman, Kendall Tau, and RBO correlations.
    """

    # Load the csv
    df = pd.read_csv(f"output/{field}_rankings.csv")

    # Take first n % of the papers
    df = df.iloc[: int(len(df) * percentile)]

    # Drop na
    df = df.dropna()

    print(df.head())

    # Convert the URLs to rankings
    for column in df.columns:
        df[column] = df[column].rank().astype(int)

    # Define the epochs
    base_epochs = ["(1980, 1990)", "(1990, 2000)", "(2000, 2010)"]
    citing_epochs_pairs = [
        ("(1990, 2000)", "(2000, 2010)"),
        ("(2000, 2010)", "(2010, 2015)"),
        ("(2010, 2015)", "(2015, 2020)"),
    ]

    # Initialize correlation tables
    spearman_table = pd.DataFrame(
        index=base_epochs,
        columns=[f"{ep1} <> {ep2}" for ep1, ep2 in citing_epochs_pairs],
    )
    kendall_table = spearman_table.copy()

    # Loop through the base epochs and citing epoch pairs to compute the correlations
    for base_epoch in base_epochs:
        for ep1, ep2 in citing_epochs_pairs:
            col1 = f"papers_from_{base_epoch}_ranked_by_citations_from_{ep1}"
            col2 = f"papers_from_{base_epoch}_ranked_by_citations_from_{ep2}"

            # Check if both columns exist in df
            if col1 in df.columns and col2 in df.columns:
                # Compute Spearman and Kendall correlations
                spearman_corr, _ = spearmanr(df[col1], df[col2])
                kendall_corr, _ = kendalltau(df[col1], df[col2])
                # Update the correlation tables
                spearman_table.at[base_epoch, f"{ep1} <> {ep2}"] = spearman_corr
                kendall_table.at[base_epoch, f"{ep1} <> {ep2}"] = kendall_corr

    # Replace NaN with a dash
    spearman_table = spearman_table.fillna("-")
    kendall_table = kendall_table.fillna("-")

    # Save tables to LaTeX files
    for table, name in zip([spearman_table, kendall_table], ["spearman", "kendall"]):
        with open(f"output/{field}_{name}_correlation_table.tex", "w") as file:
            file.write(table.to_latex())

    # Return the correlation tables
    return {"Spearman": spearman_table, "Kendall": kendall_table}


def compute_average_citation_age():
    # Pattern to match all relevant CSV files
    pattern = os.path.join("data", "byfield", '*_paper_to_stats.csv')
    
    # List to hold data from all matching files
    aggregated_data = []
    
    # Find all files matching the pattern
    for filepath in glob.glob(pattern):
        # Extract field name from the filename
        filename = os.path.basename(filepath)
        field = filename.replace('_paper_to_stats.csv', '')
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Compute the average citation age per year
        avg_citation_age_by_year = df.groupby('year')['avg_outgoing_citation_age'].mean().reset_index()
        
        # Add the field name to the DataFrame
        avg_citation_age_by_year['field'] = field
        
        # Append the results to the aggregated data list
        aggregated_data.append(avg_citation_age_by_year)
    
    # Concatenate all DataFrames in the list
    final_df = pd.concat(aggregated_data, ignore_index=True)
    
    # Reorder the columns
    final_df = final_df[['field', 'year', 'avg_outgoing_citation_age']]
    
    # Output the DataFrame to a CSV file
    output_path = os.path.join('data', 'citation_ages_by_year_and_concept.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")


def compute_volume_age_correlation():
    # Load the datasets
    volume_df = pd.read_csv('data/works_by_year_and_concept.csv')
    citation_df = pd.read_csv('data/citation_ages_by_year_and_concept.csv')
    
    # Merge the two DataFrames on 'field' and 'year'
    merged_df = pd.merge(volume_df, citation_df, on=['field', 'year'])
    
    # Define the time ranges
    time_ranges = [(1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020)]
    
    # Initialize a dictionary to store correlation results
    correlations = {}

    # Compute overall and time range-specific correlations
    for field, group_df in merged_df.groupby('field'):
        correlations[field] = {}
        
        # Overall correlation
        overall_corr = group_df[['count', 'avg_outgoing_citation_age']].corr(method='pearson').iloc[0, 1]
        correlations[field]['Overall'] = overall_corr
        
        # Time range-specific correlations
        for start_year, end_year in time_ranges:
            range_df = group_df[(group_df['year'] >= start_year) & (group_df['year'] <= end_year)]
            if not range_df.empty:
                corr = range_df[['count', 'avg_outgoing_citation_age']].corr(method='pearson').iloc[0, 1]
                correlations[field][f'{start_year}-{end_year}'] = corr
            else:
                correlations[field][f'{start_year}-{end_year}'] = 'N/A'  # In case there's no data in the time range

    return correlations

def generate_latex_table(correlations):
    # Convert the nested dictionary into a list of dictionaries for easier DataFrame creation
    data = []
    for field, time_ranges in correlations.items():
        row = {'Field': field}
        row.update(time_ranges)
        data.append(row)
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    # Set the 'Field' column as the DataFrame index
    df.set_index('Field', inplace=True)
    
    # Generate the LaTeX table
    latex_table = df.to_latex(float_format="{:0.2f}".format, escape=False)
    
    # Write to a .tex file
    with open('output/volume_age_correlation_table.tex', 'w') as f:
        f.write(latex_table)

if __name__ == "__main__":

    compute_average_citation_age()
    correlations = compute_volume_age_correlation()
    # Display the correlations
    for field, data in correlations.items():
        print(f'Field: {field}')
        for time_range, correlation in data.items():
            print(f'  {time_range}: {correlation}')

    generate_latex_table(correlations)

    # print("Start")
    # compute_rankings("NLP")
    # correlation_tables = compute_ranking_correlations("NLP", 0.5)
    # print("End")
