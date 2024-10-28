import gzip
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import io
import itertools
import json

from pep.ui import *
from pep.util import *
from pep.filter import *

import re
import copy
from typing import Dict, List, Tuple


def execute_filtering(filter_config: dict):
    """
    Execute the filtering process on gene and peptide datasets using the provided filter configuration.

    This function filters the datasets based on parameters stored in the Streamlit session state
    and updates the session state with the filtering results, including summary DataFrames and
    filtered datasets.

    Parameters
    ----------
    filter_config : dict
        Dictionary containing the filter configurations.

    Returns
    -------
    None

    Notes
    -----
    - This function depends on the `filter_datasets` function to perform the actual filtering.
    - The filtering parameters are extracted from `st.session_state` within `filter_datasets`.
    - The results are stored in `st.session_state` for later use in the application.
    """
    # Filter the datasets using the filter_datasets function
    summary_genes_df, summary_peptides_df, filtered_genes_df, filtered_peptides_df = (
        filter_datasets(
            gene_metrics_df=st.session_state.gene_metrics_df,
            peptide_metrics_df=st.session_state.peptide_metrics_df,
            filter_config=filter_config,
            component_keys=list(UI_CONFIG.keys()),
        )
    )

    # Update the Streamlit session state with the filtering results
    st.session_state.summary_genes_df = summary_genes_df
    st.session_state.summary_peptides_df = summary_peptides_df
    st.session_state.filtered_genes_df = filtered_genes_df
    st.session_state.filtered_peptides_df = filtered_peptides_df


def render_ui_form(text_config: dict, ui_config: dict, filter_config: dict):
    """
    Display the filtering form in the Streamlit app and handle user interactions.

    This function renders a form containing various filtering options for the user to select.
    Upon submission, it executes the filtering process and updates the application state.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the app, such as titles and descriptions.
    ui_config : dict
        Dictionary containing UI component configurations, including default values and options.
    filter_config : dict
        Dictionary containing the filter configurations.

    Returns
    -------
    None

    Notes
    -----
    - The form includes sections for human identity filters, strain conservation filters,
        gene expression filters, and gene homology filters.
    - When the user submits the form, the `execute_filtering` function is called to apply
        the filters to the datasets.
    - The filtering results are stored in `st.session_state`.
    """
    # Create a form for filtering candidates
    with st.form(key="filter_form"):
        # Display the human identity filters section
        show_human_identity_filters(text_config, ui_config)

        # Display the strain conservation filters section
        show_strain_conservation_filters(text_config, ui_config)

        # Display the filters for indels.
        show_indel_frequency_filters(text_config, ui_config)

        # Display the gene expression filters section
        show_gene_expression_filters(text_config, ui_config)

        # Display the gene homology filters section
        show_gene_homology_filters(text_config, ui_config)

        # Add spacing for better layout
        st.write("")
        st.write("")

        # Submit button for the form
        submit_button = st.form_submit_button(label="Filter Candidates")

        # If the submit button is clicked, execute the filtering process
        if submit_button:
            execute_filtering(filter_config)


def display_gene_results(text_config: dict):
    """
    Display the gene filtering results in the Streamlit app.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the results section.

    Returns
    -------
    None

    Notes
    -----
    This function displays the gene results including:
    - A summary of the filtering steps.
    - The number of genes retained.
    - The total number of peptides.
    - The average number of peptides per gene.
    - Allows downloading the filtering summary and candidate genes data.
    """

    def parse_percentage_value(value):
        """
        Parse the percentage value, replacing 'min' with 0 and 'max' with 100.
        """
        if isinstance(value, str):
            value = value.replace("min", "0").replace("max", "100")
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid percentage value: {value}")

    # Calculate the number of genes retained after filtering
    num_genes_retained = len(st.session_state.filtered_genes_df)

    # Calculate the total number of peptides in the retained genes
    num_total_peptides = st.session_state.filtered_genes_df["num_peptides"].sum()

    with st.container():

        st.write("\n")
        # Retrieve gene results text from the configuration
        gene_results_text = text_config["results"]["gene_results"]

        # Display the title and description paragraphs
        st.write(gene_results_text["title"])
        for paragraph in gene_results_text["description"]:
            st.write(paragraph)

        # Render the text for the filter.
        st.write(gene_results_text['title'])
        for paragraph in gene_results_text['description']:
            st.write(paragraph)
        # Expander explanations.
        if gene_results_text.get('expander'):
            with st.expander(gene_results_text['expander']['title']):
                for section in gene_results_text['expander']['content']:
                    if 'title' in section:
                        st.write(section['title'])
                    for paragraph in section['paragraphs']:
                        st.write(paragraph)
                    
        # Display the summary DataFrame with 'filter' as index and without 'order' column
        st.dataframe(
            st.session_state.summary_genes_df.set_index("filter").drop("order", axis=1)
        )

        # Display summary of the filtering rules based on session state parameters

        # Human identity gene-extrapolation rule
        if st.session_state.human_id_gene_rule == "Use mean values over gene peptides":
            st.markdown(
                "###### Human identity gene-extrapolation rule: `Mean over gene peptides`"
            )
        else:
            # Parse and handle 'min' and 'max' values for human_id_gene_pc
            human_id_gene_pc = parse_percentage_value(st.session_state.human_id_gene_pc)
            st.markdown(
                "###### Human identity gene-extrapolation rule: `Minimum percentage of gene peptides`"
            )
            st.write(
                f"- Up to **{100 - human_id_gene_pc}% of gene peptides can fail** identity and length filters"
            )

        # Strain conservation gene-extrapolation rule
        if (
            st.session_state.haplotype_gene_rule
            == "Use frequency of full exon haplotype"
        ):
            st.markdown(
                "###### Strain conservation gene-extrapolation rule: `Full gene haplotype`"
            )
        else:
            # Parse and handle 'min' and 'max' values for haplotype_gene_pc
            haplotype_gene_pc = parse_percentage_value(
                st.session_state.haplotype_gene_pc
            )
            st.markdown(
                "###### Strain conservation gene-extrapolation rule: `Minimum percentage of gene peptides`"
            )
            st.write(
                f"- At least **{haplotype_gene_pc}% of gene peptides must pass** frequency filter"
            )

        # Display statistics
        st.write("\n")
        st.markdown(f"##### Genes retained: `{num_genes_retained:,}`")
        st.write(f"##### Total number of peptides: `{num_total_peptides:,}`")

        # Calculate average number of peptides per gene, avoiding division by zero
        if num_genes_retained > 0:
            avg_peptides_per_gene = round(num_total_peptides / num_genes_retained, 2)
        else:
            avg_peptides_per_gene = 0

        # Display the average number of peptides per gene
        st.write(
            f"##### Average number of peptides per gene: `{avg_peptides_per_gene:,}`"
        )

        # Add a horizontal separator
        st.markdown("---")

        # Create columns for layout control
        lc, rc, dc, _ = st.columns([1, 1, 1, 1])
        # Download button for filtering summary
        lc.download_button(
            label="Download Filtering Summary (Genes)",
            data=dataframe_to_csv(st.session_state.summary_genes_df),
            file_name="candidate-genes-filter-summary.csv",
            mime="text/csv",
        )

        # Function to generate and download the candidate genes data
        def generate_csv_and_download_genes(ctx):
            """
            Generate the candidate genes data and provide a download button.

            Parameters
            ----------
            ctx : streamlit.delta_generator.DeltaGenerator
                The Streamlit context to display the download button.
            """
            # Select relevant columns and rename for clarity
            gene_data_df = st.session_state.filtered_genes_df[
                [
                    "gene_id",
                    "chromosome",
                    "start",
                    "end",
                    "gene_name",
                    "strand",
                    "num_exons",
                    "num_transcripts",
                    "plasmo_db_url",
                    "num_peptides",
                    "ref_3D7_haplotype",
                    "most_frequent_haplotype",
                    "ref_3D7_peptides_list",
                    "most_frequent_peptides_list"
                ]
            ].rename(columns={"hap_top_hap": "gene_haplotype"})

            # Provide a download button for the candidate genes data
            ctx.download_button(
                label="Download Table of Candidate Genes",
                data=dataframe_to_csv(gene_data_df),
                file_name="candidate-genes.csv",
                mime="text/csv",
            )

        # Button to prepare and provide the download for genes
        if rc.button("Prepare Download for Genes"):
            # Call the function to generate and display the download button
            generate_csv_and_download_genes(dc)


def display_peptide_results(text_config: dict):
    """
    Display the peptide filtering results in the Streamlit app.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the results section.

    Returns
    -------
    None

    Notes
    -----
    This function displays the peptide results including:
    - A summary of the filtering steps.
    - The number of peptides retained.
    - The number of genes involved.
    - The average number of peptides per gene.
    - Allows downloading the filtering summary and candidate peptides data.
    """
    # Calculate the number of genes involved (with at least one peptide retained)
    num_genes_retained = len(set(st.session_state.filtered_peptides_df["gene_id"]))

    # Calculate the total number of peptides retained after filtering
    num_total_peptides = len(st.session_state.filtered_peptides_df)

    with st.container():
        # Add spacing for readability
        st.write("\n")
        st.write("\n")

        # Retrieve peptide results text from the configuration
        peptide_results_text = text_config["results"]["peptide_results"]
        # Display the title and description paragraphs
        st.write(peptide_results_text["title"])
        for paragraph in peptide_results_text["description"]:
            st.write(paragraph)

        # Display the summary DataFrame with 'filter' as index and without 'order' column
        st.dataframe(
            st.session_state.summary_peptides_df.set_index("filter").drop(
                "order", axis=1
            )
        )

        # Display the number of peptides retained
        st.write(f"##### Peptides retained: `{num_total_peptides:,}`")

        # Display the number of genes involved
        st.write(
            f"##### Genes involved (at least one peptide retained): `{num_genes_retained:,}`"
        )

        # Calculate average number of peptides per gene, avoiding division by zero
        if num_genes_retained > 0:
            avg_peptides_per_gene = round(num_total_peptides / num_genes_retained, 2)
        else:
            avg_peptides_per_gene = 0

        # Display the average number of peptides per gene
        st.write(
            f"##### Average number of peptides per gene: `{avg_peptides_per_gene:,}`"
        )

        # Prepare the peptide results DataFrame for download
        peptide_results_df = st.session_state.filtered_peptides_df[
            [
                "peptide_id",
                "gene_id",
                "start",
                "end",                
                "chromosome",
                "gene_name",
                "plasmo_db_url",
                "ref_3D7_peptide",
                "most_frequent_peptide",
            ]
        ].rename(columns={"hap_top_hap": "peptide_haplotype"})

        # Determine if data should be compressed based on its size
        compress_data = len(peptide_results_df) > 20000

        # Function to generate and download the candidate peptides data
        def generate_csv_and_download_peptides(ctx):
            """
            Generate the candidate peptides data and provide a download button.

            Parameters
            ----------
            ctx : streamlit.delta_generator.DeltaGenerator
                The Streamlit context to display the download button.

            Returns
            -------
            None
            """
            # Convert the DataFrame to CSV format, compressing if necessary
            csv_data = dataframe_to_csv(peptide_results_df, compress_data)

            # Determine the file name and MIME type based on compression
            file_name = "candidate-peptides.csv" + (".gz" if compress_data else "")
            mime_type = "application/gzip" if compress_data else "text/csv"

            # Provide a download button for the candidate peptides data
            ctx.download_button(
                label="Download Table of Candidate Peptides",
                data=csv_data,
                file_name=file_name,
                mime=mime_type,
            )

        # Add a horizontal separator
        st.markdown("---")

        # Create columns for layout control
        lc, rc, dc, _ = st.columns([1, 1, 1, 1])

        # Download button for filtering summary
        lc.download_button(
            label="Download Filtering Summary (Peptides)",
            data=dataframe_to_csv(st.session_state.summary_peptides_df),
            file_name="candidate-peptides-filter-summary.csv",
            mime="text/csv",
        )

        # Button to prepare and provide the download for peptides
        if rc.button("Prepare Download for Peptides"):
            # Call the function to generate and display the download button
            generate_csv_and_download_peptides(dc)


def render_ui(text_config: dict, ui_config: dict, filter_config: dict):
    """
    Render the main user interface of the Streamlit application.

    This function sets up the UI components, initializes default values,
    executes initial filtering, and displays the results.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the app, such as titles and descriptions.
    ui_config : dict
        Dictionary containing UI component configurations, including default values and options.
    filter_config : dict
        Dictionary containing filter configurations used in the filtering process.

    Returns
    -------
    None

    Notes
    -----
    - The function initializes UI components with default values and stores them in `st.session_state`.
    - It ensures that the filtering is executed at least once to display initial results.
    - The UI form is rendered, and the filtering results are displayed.
    - Assumes that necessary functions like `execute_filtering`, `show_form`,
    `display_gene_results`, and `display_peptide_results` are defined elsewhere.
    """

    # Create columns for layout and place the logo in the center column
    col1, logo_col, col3 = st.columns([1, 4, 1])
    with logo_col:
        st.image("logo-medium.png", use_column_width=True)

    # Display the app title
    st.title(text_config["app_title"])

    # Display the app description paragraphs
    for paragraph in text_config["app_description"]:
        st.write(paragraph)

    # Initialize UI components with default values and store them in st.session_state
    for ui_key in ui_config.keys():
        component_config = ui_config[ui_key]

        # If a default value is specified, use it
        if "default" in component_config:
            st.session_state[ui_key] = component_config["default"]
        # Else, use the option at the specified index
        else:
            default_index = component_config.get(
                "index", 0
            )  # Default to index 0 if not specified
            st.session_state[ui_key] = component_config["options"][default_index]

    # Check if filtering has already been executed by looking for 'summary_genes_df' in session_state
    if "summary_genes_df" not in st.session_state:
        # Execute initial filtering to populate results
        execute_filtering(filter_config)

    # Render the UI form with filter options
    render_ui_form(text_config, ui_config, filter_config)

    # Display filtering results for genes and peptides
    display_gene_results(text_config)
    display_peptide_results(text_config)


# Main navigation
def main():

    # Load text configuration.
    TEXT_CONFIG = load_json_config("config/text.json")

    # Load filters configuration.
    FILTER_CONFIG = load_json_config("config/filters.json")

    # Load the original datasets into session state (only once).
    # These are big files, it will take a few seconds.
    if "gene_metrics_df" not in st.session_state:
        st.session_state.gene_metrics_df = pd.read_csv(
            "data/gene-metrics-filtering.csv.gz"
        )
    if "peptide_metrics_df" not in st.session_state:
        st.session_state.peptide_metrics_df = pd.read_csv(
            "data/peptide-metrics-filtering.csv.gz", low_memory=False
        )

    render_ui(TEXT_CONFIG, UI_CONFIG, FILTER_CONFIG)  # from pep.ui


if __name__ == "__main__":
    main()
