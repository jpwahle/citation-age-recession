import time
from collections import Counter, OrderedDict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyalex
import requests
from pyalex import Works

# Create data, output, and figure directories
Path("data").mkdir(exist_ok=True)
Path("data/byfield").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42
pyalex.config.email = "jan.philip.wahle@gmail.com"

general_subjects_to_urls = {
    "NLP": "https://openalex.org/C204321447",
    "ML": "https://openalex.org/C119857082",
    "Psychology": "https://openalex.org/C15744967",
    "Mathematics": "https://openalex.org/C33923547",
    "Computer science": "https://openalex.org/C41008148",
    "Sociology": "https://openalex.org/C144024400",
    "Linguistics": "https://openalex.org/C41895202",
    "AI": "https://openalex.org/C154945302",
    "Political science": "https://openalex.org/C17744445",
    "Philosophy": "https://openalex.org/C138885662",
    "Economics": "https://openalex.org/C162324750",
    "Business": "https://openalex.org/C144133560",
    "Medicine": "https://openalex.org/C71924100",
    "Biology": "https://openalex.org/C86803240",
    "Geology": "https://openalex.org/C127313418",
    "Chemistry": "https://openalex.org/C185592680",
    "Art": "https://openalex.org/C142362112",
    "Engineering": "https://openalex.org/C127413603",
    "Geography": "https://openalex.org/C205649164",
    "History": "https://openalex.org/C95457728",
    "Materials science": "https://openalex.org/C192562407",
    "Physics": "https://openalex.org/C121332964",
    "Environmental science": "https://openalex.org/C39432304",
}

# Make the above an ordered dict
general_subjects_to_urls = OrderedDict(general_subjects_to_urls)

# Load concepts from OpenAlex
concepts = pd.read_csv("data/openalex/openalex-concepts-aug-2022.csv")

nlp_subfields = concepts[
    concepts["parent_display_names"].str.contains(
        "Natural language processing", na=False
    )
]
# Capitalize the letter "c" in each nlp_subfields["openalex_id"].values
nlp_subfields.loc[:, "openalex_id"] = nlp_subfields["openalex_id"].str.replace("c", "C")

ai_subfields = concepts[
    concepts["parent_display_names"].str.contains("Artificial intelligence", na=False)
]
# Capitalize the letter "c" in each ai_subfields["openalex_id"].values
ai_subfields.loc[:, "openalex_id"] = ai_subfields["openalex_id"].str.replace("c", "C")


ml_subfields = concepts[
    concepts["parent_display_names"].str.contains("Machine learning", na=False)
]
# Capitalize the letter "c" in each ml_subfields["openalex_id"].values
ml_subfields.loc[:, "openalex_id"] = ml_subfields["openalex_id"].str.replace("c", "C")


def get_works_by_year_and_concept(year_start=1970, year_end=2023):
    """
    Filters works by year and then groups by the concept.id to get the counts by year for all concepts in general_subjects_to_urls.

    Returns:
        DataFrame: A DataFrame with counts by year for all concepts.
    """
    works_by_year_and_concept = []

    for field, url in general_subjects_to_urls.items():
        for year in range(year_start, year_end):
            concept_id = url.split("/")[-1]
            works = (
                Works()
                .filter(concepts={"id": concept_id})
                .filter(publication_year=year)
                .group_by("concept.id")
                .get(per_page=1, page=1)
            )
            print(works)
            works_by_year_and_concept.append(
                {
                    "field": works[0]["key_display_name"],
                    "year": year,
                    "count": works[0]["count"] if works else 0,
                },
            )

    return pd.DataFrame(works_by_year_and_concept)


def get_total_works_by_concept():
    """
    Counts the total number of works per concept.

    Returns:
        DataFrame: A DataFrame with total counts for all concepts.
    """
    total_works_by_concept = []

    for field, url in general_subjects_to_urls.items():
        concept_id = url.split("/")[-1]
        works = (
            Works()
            .filter(concepts={"id": concept_id})
            .group_by("concept.id")
            .get(per_page=1, page=1)
        )
        total_works_by_concept.append(
            {
                "field": works[0]["key_display_name"],
                "count": works[0]["count"] if works else 0,
            },
        )

    return pd.DataFrame(total_works_by_concept)


def get_works_by_field(concept_id, sample_size, random_seed=RANDOM_SEED):
    """
    Get the works for a given field with a specific samples size that is drawn i.i.d. (up to 10,000 works).

    Args:
        concept_id (str): The ID of the field concept to get works for.
        sample_size (int): The number of works to sample.
        random_seed (int, optional): The random seed to use for sampling. Defaults to RANDOM_SEED.

    Returns:
        list: A list of works for the given field.
    """
    sample_populated = []
    max_sample_size = 10000
    per_page = 200

    if sample_size <= max_sample_size:
        samples = (
            Works()
            .filter(concepts={"id": concept_id})
            .sample(sample_size, seed=random_seed)
        )
        page = 1
        while page <= 50:
            samples_page = samples.get(per_page=per_page, page=page)
            if not samples_page:
                break
            sample_populated.extend(samples_page)
            page += 1
    else:
        for i in range(sample_size // max_sample_size):
            if sample_size // (max_sample_size * (i + 1)) == 0:
                sample_size_current = sample_size % max_sample_size
            else:
                sample_size_current = max_sample_size

            samples = (
                Works()
                .filter(concepts={"id": concept_id})
                .sample(max_sample_size, seed=random_seed)
            )
            page = 1
            while page <= 50:
                samples_page = samples.get(per_page=per_page, page=page)
                if not samples_page:
                    break
                sample_populated.extend(samples_page)
                page += 1

    return sample_populated


def populate_incoming_cited_works(work):
    """
    Populates the incoming cited works for a given work.

    Args:
        work (dict): A dictionary representing the focal work.

    Returns:
        list: A list of dictionaries representing the works that cite the focal work.
    """
    # Get the actual works that cite the focal work.
    citing_works = []
    page = 1

    citing_works_page = Works().filter(cites=work["id"])
    while True:
        citing_works_current_page = citing_works_page.get(per_page=50, page=page)
        if not citing_works_current_page:
            break
        citing_works.extend(citing_works_current_page)
        page += 1
    return citing_works


def populate_outgoing_cited_works(work):
    """
    Populates the list of outgoing cited works for a given work.

    Args:
        work (dict): A dictionary representing a work.

    Returns:
        list: A list of cited works.
    """
    max_parallel_refs = 50
    cited_works = []
    page = 1

    if not work["referenced_works"]:
        return None
    while True:
        current_refs = work["referenced_works"][
            (page - 1) * max_parallel_refs : page * max_parallel_refs
        ]
        if not current_refs:
            break
        works_page = (
            Works()
            .filter(ids={"openalex": "|".join(current_refs)})
            .get(per_page=max_parallel_refs, page=1)
        )
        if not works_page:
            break
        cited_works.extend(works_page)
        page += 1
    return cited_works


def compute_citation_ages(focal_work, cited_or_citing_works):
    """
    Computes the age difference between the publication year of the focal work and the publication year of each cited or citing work.

    Args:
        focal_work (dict): A dictionary representing the focal work.
        cited_or_citing_works (list): A list of dictionaries representing the cited or citing works.

    Returns:
        list: A list of integers representing the age difference between the publication year of the focal work and the publication year of each cited or citing work.
    """

    return [
        abs(cited_or_citing_work["publication_year"] - focal_work["publication_year"])
        for cited_or_citing_work in cited_or_citing_works
    ]


def get_top_level_parent(ror_id):
    """
    Recursively gets the top-level parent of an organization with the given ROR ID.

    Args:
        ror_id (str): The ROR ID of the organization.

    Returns:
        Tuple[str, str]: A tuple containing the ROR ID and name of the top-level parent organization.
        If no parent relationship is found, returns (None, None).
    """
    ror_url = f"https://api.ror.org/organizations/{ror_id}"
    response = requests.get(ror_url)

    if response.status_code == 200:
        institution_data = response.json()
        relationships = institution_data.get("relationships")

        for relationship in relationships:
            if relationship.get("type") == "parent":
                parent_ror_id = relationship.get("id")
                return get_top_level_parent(
                    parent_ror_id
                )  # Recursively get top-level parent

        return ror_id, institution_data.get(
            "name"
        )  # No parent relationship found, this is the top-level parent

    return None, None


def compute_field_assignment(works, citation_ages, field_scores_threshold):
    """
    Computes the field assignment for a given set of works and citation ages.

    Args:
        works (list): A list of works, where each work is a dictionary containing
            information about the work, including its concepts.
        citation_ages (list): A list of citation ages, where each age corresponds
            to a work in the `works` list.
        field_scores_threshold (float): The threshold score for a field to be considered
            in the field assignment.

    Returns:
        tuple: A tuple containing four elements:
            - A list of fields, where each field is a list of field names for a given work.
            - The Gini-Simpson index for the fields.
            - A list of citation ages for works in the "Computer Science" field.
            - A list of citation ages for works not in the "Computer Science" field.
    """
    fields = []
    ages_cs = []
    ages_non_cs = []

    for work, citation_age in zip(works, citation_ages):
        if work["concepts"]:
            # Get all fields that have > field_scores_threshold score and are level 0 or Linguistics
            work_fields = [
                field["display_name"]
                for field in work["concepts"]
                if float(field["score"]) > field_scores_threshold
                and (
                    field["level"] == 0
                    or field["id"] == general_subjects_to_urls["Linguistics"]
                )
                and field["id"] != general_subjects_to_urls["ML"]
                and field["id"] != general_subjects_to_urls["AI"]
                and field["id"] != general_subjects_to_urls["NLP"]
            ]
            # Add the fields to the list of incoming fields as a sublist
            fields.append(work_fields)
            # Get the field with the largest "score" that is a level 0 field and Linguistics which is not a level 0 field here
            work_field = max(
                work["concepts"],
                key=lambda x: x["score"]
                if x["level"] == 0 or x["id"] == general_subjects_to_urls["Linguistics"]
                else 0,
            )
            # Categorize into CS and non-CS citation ages
            if work_field["id"] == general_subjects_to_urls["Computer science"]:
                ages_cs.append(citation_age)
            else:
                ages_non_cs.append(citation_age)

    if work_fields_for_gini := [item for sublist in fields for item in sublist]:
        # Compute Gini-Simpson index for fields.
        cfdi = gini_simpson_index(work_fields_for_gini)
    else:
        cfdi = None

    return fields, cfdi, ages_cs, ages_non_cs


def gini_index(frequencies):
    """
    Calculate the Gini index of a list of frequencies.

    Parameters:
    frequencies (list): A list of integers or strings representing the frequencies.

    Returns:
    float: The Gini index of the frequencies. Returns None if the list is empty or the mean of the frequencies is zero.
    """
    if type(frequencies[0]) == str:
        frequencies = list(Counter(frequencies).values())
    frequencies = np.array(frequencies)
    if len(frequencies) == 0 or np.mean(frequencies) == 0:
        return None
    total = sum(
        np.sum(np.abs(xi - frequencies[i:])) for i, xi in enumerate(frequencies[:-1], 1)
    )
    return total / (len(frequencies) ** 2 * np.mean(frequencies))


def gini_simpson_index(frequencies):
    """
    Calculates the Gini-Simpson index for a list of frequencies.

    The Gini-Simpson index is a measure of diversity that takes into account the
    number of unique items and their relative frequencies. It ranges from 0 to 1,
    where 0 indicates maximum diversity (all items have equal frequency) and 1
    indicates minimum diversity (only one item has non-zero frequency).

    Args:
        frequencies (list): A list of integers representing the frequency of each item.

    Returns:
        float: The Gini-Simpson index for the given frequencies.

    """
    # Count occurrences of each unique citation age
    counts = Counter(frequencies)
    total_citations = len(frequencies)
    # Compute proportion for each citation frequency and square it
    proportions_squared = [(count / total_citations) ** 2 for count in counts.values()]
    return 1 - sum(proportions_squared)


def test_main_functions():
    """
    This function tests the main functions of the module. It calls the following functions:
    - get_works_by_field
    - populate_incoming_cited_works
    - compute_citation_ages
    - get_top_level_parent
    - _test_gini

    It prints the results of each function call.
    """
    works = get_works_by_field(
        concept_id="https://openalex.org/c203005215", sample_size=500
    )

    incoming_works = populate_incoming_cited_works(works[50])
    print(incoming_works)

    citation_ages = compute_citation_ages(works[0], incoming_works)
    print(citation_ages)

    top_level_parent = get_top_level_parent(
        "https://ror.org/02388em19"
    )  # Facebook (Israel)
    print(top_level_parent)

    citation_ages = [1, 1, 2, 2, 3, 3, 4, 4]
    _test_gini(
        citation_ages,
        "Gini-Simpson Index for Citation Ages: ",
        "Gini Index for Citation Ages: ",
    )
    citation_ages = [1, 20, 50]
    _test_gini(
        citation_ages,
        "Gini-Simpson Index for Citation Ages: ",
        "Gini Index for Citation Ages: ",
    )
    field_citations = [
        "Computer science",
        "Computer science",
        "Computer science",
        "Computer science",
        "Psychology",
    ]
    _test_gini(
        field_citations,
        "Gini-Simpson Index for Fields: ",
        "Gini Index for Fields: ",
    )
    field_citations = [
        "Computer science",
        "Philosophy",
        "Psychology",
        "Linguistics",
        "Psychology",
    ]
    _test_gini(
        field_citations,
        "Gini-Simpson Index for Fields: ",
        "Gini Index for Fields: ",
    )


def _test_gini(frequencies, name_age, name_fieds):
    """
    Calculates and prints the Gini-Simpson index and Gini index for the given frequencies.

    Parameters:
    frequencies (list): A list of integers representing the frequency of each category.
    name_age (str): A string representing the name of the age category.
    name_fieds (str): A string representing the name of the fields category.
    """
    result = gini_simpson_index(frequencies)
    print(f"{name_age}{result}")
    result = gini_index(frequencies)
    print(f"{name_fieds}{result}")


def get_seconds_until_retry(retry_after_string):
    """
    Calculates the number of seconds until a retry can be attempted based on the Retry-After header value.

    Args:
        retry_after_string (str): The value of the Retry-After header.

    Returns:
        float: The number of seconds until a retry can be attempted.
    """
    # Parse the date-time string from the Retry-After header
    retry_time = datetime.strptime(retry_after_string, "%a, %d %b %Y %H:%M:%S %Z")

    # Compute the difference between the retry_time and the current time
    delta = retry_time - datetime.utcnow()

    return delta.total_seconds()


def process_work(work, sample_for_field, field_scores_threshold=0.5):
    """
    Process a given work and extract various features such as citation counts, venue, year, institutions, and subfields.

    Args:
        work (dict): A dictionary containing information about the work to be processed.
        sample_for_field (str): A string indicating the sample field to extract subfields for. Can be "NLP" or "ML".
        field_scores_threshold (float, optional): A float indicating the threshold for subfield scores. Defaults to 0.5.

    Returns:
        stats (list): A list of statistics extracted from the work.
        institutions (list): A list of institutions associated with the work.
        subfields (list): A list of subfields associated with the work, based on the sample_for_field parameter.
    """
    retries = 20

    while retries > 0:
        try:
            stats = []
            institutions = []
            incoming_works = populate_incoming_cited_works(work)

            incoming_citation_ages = (
                incoming_fields
            ) = (
                incoming_cfdi
            ) = incoming_citation_ages_cs = incoming_citation_ages_non_cs = None
            if incoming_works:
                incoming_citation_ages = compute_citation_ages(work, incoming_works)
                incoming_cad = gini_index(incoming_citation_ages)
                (
                    incoming_fields,
                    incoming_cfdi,
                    incoming_citation_ages_cs,
                    incoming_citation_ages_non_cs,
                ) = compute_field_assignment(
                    incoming_works,
                    incoming_citation_ages,
                    field_scores_threshold,
                )

            outgoing_works = populate_outgoing_cited_works(work)
            outgoing_citation_ages = (
                outgoing_fields
            ) = (
                outgoing_cfdi
            ) = outgoing_citation_ages_cs = outgoing_citation_ages_non_cs = None
            if outgoing_works:
                outgoing_citation_ages = compute_citation_ages(work, outgoing_works)
                outgoing_cad = gini_index(outgoing_citation_ages)
                (
                    outgoing_fields,
                    outgoing_cfdi,
                    outgoing_citation_ages_cs,
                    outgoing_citation_ages_non_cs,
                ) = compute_field_assignment(
                    outgoing_works,
                    outgoing_citation_ages,
                    field_scores_threshold,
                )

            # Calculate the citation counts
            incoming_citation_count = work["cited_by_count"]
            outgoing_citation_count = len(work["referenced_works"])

            # Get the venue
            venue_id = None
            venue_name = None
            if (
                work["primary_location"] != None
                and work["primary_location"]["source"] != None
            ):
                venue_id = work["primary_location"]["source"]["id"]
                venue_name = work["primary_location"]["source"]["display_name"]

            # Get the year
            year = work["publication_year"]

            # Get institutions
            institutions_distinct_count = 0
            if "institutions_distinct_count" in work.keys():
                institutions_distinct_count = (
                    work["institutions_distinct_count"]
                    if work["institutions_distinct_count"] != None
                    else 0
                )

            for author in work["authorships"]:
                for institution in author["institutions"]:
                    institution_id = institution["id"]
                    institution_name = institution["display_name"]
                    institution_ror = institution["ror"]
                    institution_country = institution["country_code"]
                    institution_type = institution["type"]

                    parent_institution_id = institution_id
                    parent_institution_name = institution_name

                    if institution_ror:
                        (
                            parent_institution_id,
                            parent_institution_name,
                        ) = get_top_level_parent(institution_ror)

                    institutions.append(
                        {
                            "institution_id": institution_id,
                            "institution_name": institution_name,
                            "paper_id": work["id"],
                            "institution_ror": institution_ror,
                            "institution_country": institution_country,
                            "type": institution_type,
                            "parent_institution_id": parent_institution_id,
                            "parent_institution_name": parent_institution_name,
                        }
                    )

            subfields = None
            if sample_for_field == "NLP":
                subfields = [
                    {
                        "paper_id": work["id"],
                        "subfield_id": field["id"],
                        "subfield_name": field["display_name"],
                        "subfield_score": field["score"],
                    }
                    for field in work["concepts"]
                    if float(field["score"]) > field_scores_threshold
                    and field["id"] in nlp_subfields["openalex_id"].values
                ]
            elif sample_for_field == "ML":
                subfields = [
                    {
                        "paper_id": work["id"],
                        "subfield_id": field["id"],
                        "subfield_name": field["display_name"],
                        "subfield_score": field["score"],
                    }
                    for field in work["concepts"]
                    if float(field["score"]) > field_scores_threshold
                    and field["id"] in ml_subfields["openalex_id"].values
                ]
            elif sample_for_field == "AI":
                subfields = [
                    {
                        "paper_id": work["id"],
                        "subfield_id": field["id"],
                        "subfield_name": field["display_name"],
                        "subfield_score": field["score"],
                    }
                    for field in work["concepts"]
                    if float(field["score"]) > field_scores_threshold
                    and field["id"] in ai_subfields["openalex_id"].values
                ]

            fields = [
                {
                    "paper_id": work["id"],
                    "field_id": field["id"],
                    "field_name": field["display_name"],
                    "field_score": field["score"],
                }
                for field in work["concepts"]
                if float(field["score"]) > field_scores_threshold
                and field["id"] in general_subjects_to_urls.values()
            ]
            grants = [
                {
                    "paper_id": work["id"],
                    "award_id": grant["award_id"],
                    "funder_name": grant["funder_display_name"],
                    "funder_id": grant["funder"],
                }
                for grant in work["grants"]
            ]
            # Store the stats
            stats = {
                "paper_id": work["id"],
                "title": work["title"],
                "year": year,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "incoming_citation_ages": incoming_citation_ages,
                "outgoing_citation_ages": outgoing_citation_ages,
                "counts_by_year": work["counts_by_year"],
                "avg_incoming_citation_age": sum(incoming_citation_ages)
                / len(incoming_citation_ages)
                if incoming_citation_ages
                else None,
                "avg_outgoing_citation_age": sum(outgoing_citation_ages)
                / len(outgoing_citation_ages)
                if outgoing_citation_ages
                else None,
                "avg_incoming_citation_age_cs": sum(incoming_citation_ages_cs)
                / len(incoming_citation_ages_cs)
                if incoming_citation_ages_cs
                else None,
                "avg_incoming_citation_age_non_cs": sum(incoming_citation_ages_non_cs)
                / len(incoming_citation_ages_non_cs)
                if incoming_citation_ages_non_cs
                else None,
                "avg_outgoing_citation_age_cs": sum(outgoing_citation_ages_cs)
                / len(outgoing_citation_ages_cs)
                if outgoing_citation_ages_cs
                else None,
                "avg_outgoing_citation_age_non_cs": sum(outgoing_citation_ages_non_cs)
                / len(outgoing_citation_ages_non_cs)
                if outgoing_citation_ages_non_cs
                else None,
                "incoming_citation_count": incoming_citation_count,
                "outgoing_citation_count": outgoing_citation_count,
                "incoming_cad": incoming_cad
                if incoming_citation_ages is not None
                else None,
                "outgoing_cad": outgoing_cad
                if outgoing_citation_ages is not None
                else None,
                "incoming_fields": incoming_fields or None,
                "outgoing_fields": outgoing_fields or None,
                "incoming_cfdi": incoming_cfdi if incoming_cfdi is not None else None,
                "outgoing_cfdi": outgoing_cfdi if outgoing_cfdi is not None else None,
            }
            return stats, institutions, grants, fields, subfields

        except requests.exceptions.HTTPError as error_e:
            wait_time = (
                get_seconds_until_retry(retry_after)
                if (retry_after := error_e.response.headers.get("Retry-After"))
                else 0.5
            )
            if wait_time > 0:
                time.sleep(wait_time)
            retries -= 1
        except Exception as error_general:
            print(error_general)
    # If retries have run out:
    if retries == 0:
        print("Max retries reached. Exiting...")
        return None, None, None, None, None


def worker(args):
    """
    Process the given works and sample for a field.

    Args:
        args (tuple): A tuple containing the current works and sample for a field.

    Returns:
        The result of processing the given works and sample for a field.
    """
    current_works, sample_for_field = args
    return process_work(current_works, sample_for_field)


def compute_per_paper_stats(
    concept_id, field, sample_size=50000, random_seed=RANDOM_SEED
):
    """
    Compute statistics for a given field and concept ID, using a sample of papers.

    Args:
        concept_id (str): The ID of the concept to use for filtering papers.
        field (str): The name of the field to compute statistics for.
        sample_size (int, optional): The number of papers to sample. Defaults to 50000.
        random_seed (int, optional): The random seed to use for sampling. Defaults to RANDOM_SEED.

    Returns:
        Tuple: A tuple containing the following lists:
            - paper_to_stats: A list of dictionaries, where each dictionary contains statistics for a paper.
            - papers_to_institutions: A list of tuples, where each tuple contains a paper ID and an institution name.
            - papers_to_grants: A list of tuples, where each tuple contains a paper ID and a grant ID.
            - papers_to_fields: A list of tuples, where each tuple contains a paper ID and a field name.
            - papers_to_subfields: A list of tuples, where each tuple contains a paper ID and a subfield name.
    """
    paper_to_stats = []
    papers_to_institutions = []
    papers_to_grants = []
    papers_to_fields = []

    page_size = 200
    parallel_requests = 3

    nlp_works = get_works_by_field(concept_id=concept_id, sample_size=sample_size)

    print("Number of works: ", len(nlp_works))

    all_results = []

    # for work in tqdm(nlp_works):
    #     stats, institutions, grants, fields = process_work(work)
    #     all_results.append((stats, institutions, grants, fields))

    with Pool(parallel_requests) as p:
        for page in range(1, sample_size // page_size + 1):
            start_time = time.time()
            current_works_list = nlp_works[(page - 1) * page_size : page * page_size]
            args_for_worker = [(dict(work), field) for work in current_works_list]
            results = p.map(worker, args_for_worker)
            all_results.extend(results)
            print(
                f"Finished page {page} of {sample_size // page_size} in"
                f" {round(time.time() - start_time, 2)} seconds."
            )

    # After gathering all the results, split them into separate lists:
    paper_to_stats = [item[0] for item in all_results]
    papers_to_institutions = [
        item
        for sublist in [
            item[1]
            for item in all_results
            if item and len(item) > 1 and item[1] is not None
        ]
        for item in sublist
        if item is not None
    ]
    papers_to_grants = [
        item
        for sublist in [
            item[2]
            for item in all_results
            if item and len(item) > 2 and item[2] is not None
        ]
        for item in sublist
        if item is not None
    ]
    papers_to_fields = [
        item
        for sublist in [
            item[3]
            for item in all_results
            if item and len(item) > 3 and item[3] is not None
        ]
        for item in sublist
        if item is not None
    ]
    papers_to_subfields = [
        item
        for sublist in [
            item[4]
            for item in all_results
            if item and len(item) > 4 and item[4] is not None
        ]
        for item in sublist
        if item is not None
    ]

    return (
        paper_to_stats,
        papers_to_institutions,
        papers_to_grants,
        papers_to_fields,
        papers_to_subfields,
    )


if __name__ == "__main__":
    overwrite_files = False # Set to True to overwrite existing files
    perc_sample_size = 0.01 # 1% of the total sample size

    works_by_concept = get_total_works_by_concept()
    works_by_concept = works_by_concept.sort_values(by="count", ascending=False)
    works_by_concept.to_csv("data/works_by_concept.csv", index=False)
    with open("data/works_by_concept.tex", "w") as tf:
        tf.write(
            works_by_concept.to_latex(
                index=False,
                caption=f"Number of papers per field for all {len(works_by_concept)} fields.",
                formatters={"count": "{:,.0f}".format},
            )
        )

    # works_by_year_and_concept = get_works_by_year_and_concept()
    # works_by_year_and_concept.to_csv("data/works_by_year_and_concept.csv", index=False)

    print("Testing main functions...")
    test_main_functions()

    for field, _ in general_subjects_to_urls.items():
        papers_to_stats_path = Path(f"data/byfield/{field}_paper_to_stats.csv")
        papers_to_institutions_path = Path(
            f"data/byfield/{field}_paper_to_institutions.csv"
        )
        papers_to_grants_path = Path(f"data/byfield/{field}_paper_to_grants.csv")
        papers_to_fields_path = Path(f"data/byfield/{field}_paper_to_fields.csv")
        papers_to_subfields_path = Path(f"data/byfield/{field}_paper_to_subfields.csv")

        if (
            not overwrite_files
            and papers_to_stats_path.is_file()
            and papers_to_institutions_path.is_file()
            and papers_to_grants_path.is_file()
            and papers_to_fields_path.is_file()
            and papers_to_subfields_path.is_file()
        ):
            print(f"Skipping {field} because the files already exist.")
            continue

        print(f"Requesting {field}")

        # Quick and dirty mapping of the non-abbreviated fields for works_by_concept
        works_by_concept_field = field
        if field == "NLP":
            works_by_concept_field = "Natural language processing"
        elif field == "ML":
            works_by_concept_field = "Machine learning"
        elif field == "AI":
            works_by_concept_field = "Artificial intelligence"

        (
            paper_to_stats,
            papers_to_institutions,
            papers_to_grants,
            papers_to_fields,
            papers_to_subfields,
        ) = compute_per_paper_stats(general_subjects_to_urls[field], field, sample_size=int(perc_sample_size * works_by_concept[works_by_concept["field"] == works_by_concept_field]["count"].values[0]))

        print(paper_to_stats is None)

        paper_to_stats_df = pd.DataFrame(paper_to_stats)
        papers_to_institutions_df = pd.DataFrame(papers_to_institutions)
        papers_to_grants_df = pd.DataFrame(papers_to_grants)
        papers_to_fields_df = pd.DataFrame(papers_to_fields)
        papers_to_subfields_df = pd.DataFrame(papers_to_subfields)

        paper_to_stats_df.to_csv(papers_to_stats_path, index=False)
        papers_to_institutions_df.to_csv(papers_to_institutions_path, index=False)
        papers_to_grants_df.to_csv(papers_to_grants_path, index=False)
        papers_to_fields_df.to_csv(papers_to_fields_path, index=False)
        papers_to_subfields_df.to_csv(papers_to_subfields_path, index=False)
