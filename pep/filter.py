import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import streamlit as st
import copy
import re


def call_expression_data(
    df: pd.DataFrame,
    stage: str,
    cutoff: float,
    min_num_replicates: int,
    total_replicates: int
) -> pd.Series:
    """
    Determine whether each gene meets the expression criteria across replicates.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing gene expression data.
    stage : str
        The expression stage to consider (used to select relevant columns).
    cutoff : float
        The CPM (Counts Per Million) cutoff value; genes must meet or exceed this value.
    min_num_replicates : int
        The minimum number of replicates that must meet or exceed the cutoff.
    total_replicates : int
        The total number of replicates available.

    Returns
    -------
    pandas.Series
        A boolean Series indicating whether each gene meets the expression criteria.
        True if the gene meets the criteria, False otherwise.
    """
    # Generate the list of column names for the specified stage and replicates
    columns = [f'exp_{stage}_cpm_r{i+1}' for i in range(total_replicates)]
    expr_data = df[columns]

    # Create a boolean DataFrame where True indicates the value meets or exceeds the cutoff
    meets_cutoff = expr_data >= cutoff

    # Count the number of replicates meeting the cutoff for each gene
    num_replicates_meeting_cutoff = meets_cutoff.sum(axis=1)

    # Determine if the number of replicates meeting the cutoff satisfies the minimum required
    result = num_replicates_meeting_cutoff >= min_num_replicates
    return result


def apply_filters_sequentially(
    df: pd.DataFrame,
    filters: List[Union[Dict, List[Dict]]],
    only_summary: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[pd.DataFrame]]]:
    """
    Apply a sequence of filters to a DataFrame, returning the step-by-step 
    filtered DataFrames and a summary.

    This function applies a list of filters (which can include nested filters) sequentially to a DataFrame.
    Each filter can be a dictionary specifying a filtering condition, or a list of such dictionaries.
    Filters within a nested list are combined using logical AND.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be filtered.
    filters : list of dict or list of list of dict
        A list of filter specifications or nested lists of filter specifications.
        Each filter specification is a dictionary with the following keys:
            - 'label': str
                A string label describing the filter.
            - 'column': str
                The column name in the DataFrame to apply the filter on.
            - 'op': str
                The operation as a string ('>=', '<=', '==', '>', '<', '!=').
            - 'value': any
                The value to compare the column against.
    only_summary : bool, optional
        If True, only return the summary DataFrame. Default is False.

    Returns
    -------
    pandas.DataFrame or tuple
        If only_summary is True, returns:
            summary_df : pandas.DataFrame
                A DataFrame containing the filter number, description, remaining rows,
                and fraction of data remaining.
        Else, returns:
            summary_df : pandas.DataFrame
                Same as above.
            filtered_dfs : list of pandas.DataFrame
                A list of DataFrames resulting from applying each filter.

    Raises
    ------
    ValueError
        If an unsupported filter operation is encountered.

    Notes
    -----
    This function is designed to work with a list of filter specifications, possibly nested.
    Filters are applied sequentially, and nested filters (lists of dictionaries) are combined using logical AND before applying.

    Example
    -------
    Given a DataFrame `df` and a list of filters:

    filters = [
        {'label': 'Filter A', 'column': 'col1', 'op': '>=', 'value': 10},
        [
            {'label': 'Filter B1', 'column': 'col2', 'op': '<=', 'value': 5},
            {'label': 'Filter B2', 'column': 'col3', 'op': '==', 'value': 'Yes'}
        ],
        {'label': 'Filter C', 'column': 'col4', 'op': '!=', 'value': 100}
    ]

    This will apply Filter A, then Filters B1 and B2 combined with AND, then Filter C.
    """
    
    # Initialize a list to store the filtered DataFrames after each filter is applied
    filtered_dfs = []
    # Initialize a list to store summary information about each filtering step
    summary_data = []

    # Record the initial state before any filters are applied
    summary_data.append({
        'order': 0,                   # The sequence number of the filter (0 for initial dataset)
        'filter': 'Initial Dataset',  # Description of the filter
        'remaining': len(df),         # Number of rows remaining after the filter
        'fraction': 1.0               # Fraction of data remaining (1.0 before any filters)
    })

    # Start with the original DataFrame
    current_df = df.copy()

    # Iterate over the list of filters
    for i, filter_group in enumerate(filters):
        # Ensure filter_group is a list of filters
        if isinstance(filter_group, dict):
            filter_group = [filter_group]

        # Initialize a list to store descriptions of each individual filter in the group
        combined_filter_description = []
        # Start with a condition that is True for all rows
        combined_condition = pd.Series(True, index=current_df.index)

        # Iterate over each filter in the group
        for filter_dict in filter_group:
            # Extract filter parameters
            label = filter_dict['label']
            column = filter_dict['column']
            op = filter_dict['op']
            value = filter_dict['value']

            # Ensure the column exists in the DataFrame
            if column not in current_df.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame.")

            # Create a filter condition based on the operation
            if op == '>=':
                condition = current_df[column] >= value
            elif op == '<=':
                condition = current_df[column] <= value
            elif op == '==':
                condition = current_df[column] == value
            elif op == '>':
                condition = current_df[column] > value
            elif op == '<':
                condition = current_df[column] < value
            elif op == '!=':
                condition = current_df[column] != value
            else:
                # Raise an error if the operation is unsupported
                raise ValueError(f"Unsupported filter operation: {op}")

            # Combine the condition with the existing conditions using logical AND
            combined_condition &= condition

            # Create a description for this individual filter
            value_repr = f"'{value}'" if isinstance(value, str) else value
            combined_filter_description.append(f"{label} {op} {value_repr}")

        # Join all individual filter descriptions with ' AND ' for the summary
        filter_description = ' AND '.join(combined_filter_description)

        # Apply the combined filter to the current DataFrame
        filtered_df = current_df[combined_condition]

        # Store the filtered DataFrame
        filtered_dfs.append(filtered_df)

        # Update the current DataFrame to the filtered result for the next iteration
        current_df = filtered_df.copy()

        # Calculate the fraction of data remaining after applying the filter
        fraction_remaining = len(filtered_df) / len(df) if len(df) > 0 else 0

        # Store summary information about this filtering step
        summary_data.append({
            'order': i + 1,                 # The sequence number of the filter
            'filter': filter_description,   # Description of the filter(s) applied
            'remaining': len(filtered_df),  # Number of rows remaining after the filter
            'fraction': fraction_remaining  # Fraction of data remaining
        })

    # Create a summary DataFrame from the collected summary data
    summary_df = pd.DataFrame(summary_data)

    # If only_summary is True, return only the summary DataFrame
    if only_summary:
        return summary_df

    # Otherwise, return both the summary DataFrame and the list of filtered DataFrames
    return summary_df, filtered_dfs


def filter_datasets(
    gene_metrics_df: pd.DataFrame, 
    peptide_metrics_df: pd.DataFrame,
    filter_config: Dict,
    component_keys: List
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filters the gene and peptide datasets based on various filtering criteria
    defined in the Streamlit session state.

    Parameters
    ----------
    gene_metrics_df : pd.DataFrame
        DataFrame containing gene metrics.
    peptide_metrics_df : pd.DataFrame
        DataFrame containing peptide metrics.
    filter_config : dict
        Dictionary containing filter configurations.
    component_keys: List
        The keys of all components registered in the UI (and global state).

    Returns
    -------
    gene_summary_df : pd.DataFrame
        Summary DataFrame of the filtering steps for genes.
    peptide_summary_df : pd.DataFrame
        Summary DataFrame of the filtering steps for peptides.
    filtered_genes_df : pd.DataFrame
        Filtered gene metrics DataFrame.
    filtered_peptides_df : pd.DataFrame
        Filtered peptide metrics DataFrame.

    Raises
    ------
    KeyError
        If a required key is not found in the Streamlit session state.

    Notes
    -----
    This function extracts filtering parameters from the Streamlit session state
    and applies filters to the gene and peptide datasets accordingly.
    """
    # List of required keys to extract from Streamlit session state

    # Extract parameters from Streamlit session state
    missing_keys = [key for key in component_keys if key not in st.session_state]
    if missing_keys:
        raise KeyError(f"Missing required parameter(s) in st.session_state: {', '.join(missing_keys)}")
    
    # Unpack parameters from session state
    p = dict()
    for k in component_keys:
        p[k] = st.session_state[k]
        
    # Begin filtering process
    # Get all the filters from the filter configuration
    filters = filter_config['filters']  
    filters_to_apply = []
    
    # Create copies of the human identity filters and update their values
    human_id_filter = filters['f_human_identity_filter'].copy()
    human_id_filter['value'] = p['identity_percent']

    human_length_filter = filters['f_human_length_filter'].copy()
    human_length_filter['value'] = p['alignment_length']

    # Adjust column names based on the gene rule for human identity
    if p['human_id_gene_rule'] == 'Use minimum percentage of peptides':
        # Append percentile suffix to column names for peptide percentage
        human_id_filter['column'] += f'_p{p["human_id_gene_pc"]:02}'
        human_length_filter['column'] += f'_p{p["human_id_gene_pc"]:02}'

    # Compose the list of filters to apply
    # Determine if human identity filters should be applied jointly or sequentially
    if p['human_id_joint_rule'] == "Joint (use both conditions in same filter)":
        # Combine human identity filters into a nested list for joint application
        filters_to_apply.append([human_id_filter, human_length_filter])
    else:
        # Apply human identity filters sequentially
        filters_to_apply.append(human_id_filter)
        filters_to_apply.append(human_length_filter)

    # Select the appropriate strain conservation filter based on haplotype type
    if p['haplotype_type'] == 'Identical to 3D7':
        # Use the filter for the 3D7 haplotype frequency
        conservation_filter = filters['f_strain_conservation_3D7'].copy()
    else:
        # Use the filter for any haplotype frequency
        conservation_filter = filters['f_strain_conservation_any_hap'].copy()

    # Adjust the column name based on the haplotype gene rule
    if p['haplotype_gene_rule'] == 'Use minimum percentage of peptides':
        # Append percentile suffix to column name for peptide percentage
        conservation_filter['column'] += f'_p{(100 - p["haplotype_gene_pc"]):02}'
    
    # Update the value for the strain conservation filter
    conservation_filter['value'] = p['strain_conservation']
    filters_to_apply.append(conservation_filter)

    # Select the required indel filter.
    # This time we have a different filter per column.
    indels_filter = None
    if p['indel_type'] == 'Any' and p['indel_frameshfits'] == 'Any':
        indels_filter = filters['f_indels_freq_any'].copy()
    elif p['indel_type'] == 'Only Insertions' and p['indel_frameshfits'] == 'Any':
        indels_filter = filters['f_indels_freq_insertions'].copy()
    elif p['indel_type'] == 'Only Deletions' and p['indel_frameshfits'] == 'Any':
        indels_filter = filters['f_indels_freq_deletions'].copy()

    elif p['indel_type'] == 'Any' and p['indel_frameshfits'] == 'Only Frameshifts':
        indels_filter = filters['f_indels_freq_any_frameshifts'].copy()
    elif p['indel_type'] == 'Only Insertions' and p['indel_frameshfits'] == 'Only Frameshifts':
        indels_filter = filters['f_indels_freq_insertions_frameshifts'].copy()
    elif p['indel_type'] == 'Only Deletions' and p['indel_frameshfits'] == 'Only Frameshifts':
        indels_filter = filters['f_indels_freq_deletions_frameshifts'].copy() 
    else:
        raise ValueError('Unknown indel filter configuration')

    # Set the filtering value (frequency).
    indels_filter['value'] = p['indel_frequency']
    
    # By default we use peptide percentage for filtering.
    # This suffix is removed by the peptide filtering funciton.
    indels_filter['column'] += f'_p{(p["indel_gene_pc"]):02}'
    filters_to_apply.append(indels_filter)
    
    # Determine the minimum number of replicates required based on the expression rule
    if p['expression_rule'] == "At least one replicate":
        min_num_replicates = 1
    elif p['expression_rule'] == "At least two replicates":
        min_num_replicates = 2
    elif p['expression_rule'] == "All replicates":
        min_num_replicates = 3
    else:
        raise ValueError(f'Unknown expression rule: {p["expression_rule"]}')

    # Add expression data columns to gene and peptide DataFrames for liver stages
    stages = ['d2', 'd4', 'd5', 'd6']
    for stage in stages:
        gene_metrics_df[f'exp_expressed_{stage}'] = call_expression_data(
            gene_metrics_df,
            stage=stage,
            cutoff=p['cpm_cutoff'],
            min_num_replicates=min_num_replicates,
            total_replicates=3
        )
        peptide_metrics_df[f'exp_expressed_{stage}'] = call_expression_data(
            peptide_metrics_df,
            stage=stage,
            cutoff=p['cpm_cutoff'],
            min_num_replicates=min_num_replicates,
            total_replicates=3
        )

    # Adjust minimum number of replicates for the sporozoite stage
    if p['expression_rule'] == "All replicates":
        min_num_replicates_sporozoite = 5  # All five replicates
    else:
        min_num_replicates_sporozoite = min_num_replicates  # Same as for liver stages

    # Add expression data columns for the sporozoite stage
    gene_metrics_df['exp_expressed_sporozoite'] = call_expression_data(
        gene_metrics_df,
        stage='sporozoite',
        cutoff=p['cpm_cutoff'],
        min_num_replicates=min_num_replicates_sporozoite,
        total_replicates=5
    )
    peptide_metrics_df['exp_expressed_sporozoite'] = call_expression_data(
        peptide_metrics_df,
        stage='sporozoite',
        cutoff=p['cpm_cutoff'],
        min_num_replicates=min_num_replicates_sporozoite,
        total_replicates=5
    )

    # Map expression stages to their corresponding filters
    expression_filters = {
        'Liver Stage Day 2': filters['f_liver_expression_d2'],
        'Liver Stage Day 4': filters['f_liver_expression_d4'],
        'Liver Stage Day 5': filters['f_liver_expression_d5'],
        'Liver Stage Day 6': filters['f_liver_expression_d6'],
        'Sporozoite Stage': filters['f_liver_expression_sporozoite']
    }

    # Add selected expression filters to the list of filters to apply
    for stage in p['expression_stage']:
        if stage in expression_filters:
            filters_to_apply.append(expression_filters[stage])
        else:
            raise ValueError(f"Unknown expression stage: {stage}")

    # Map homology species to their corresponding filters
    homology_filters = {
        'P. berghei': filters['f_pb_homology'],
        'P. vivax': filters['f_pv_homology'],
        'P. knowlesi': filters['f_pk_homology']
    }

    # Add selected homology filters to the list of filters to apply
    for species in p['homology_species']:
        if species in homology_filters:
            filters_to_apply.append(homology_filters[species])
        else:
            raise ValueError(f"Unknown homology species: {species}")

    # Apply filters to the gene metrics DataFrame
    gene_summary_df, filtered_genes_dfs = apply_filters_sequentially(
        gene_metrics_df,
        filters=filters_to_apply,
        only_summary=False
    )
    # Extract the final filtered gene DataFrame
    filtered_genes_df = filtered_genes_dfs[-1] # type: ignore

    # Adjust filters for peptides to match the appropriate columns
    # Deep copy the filters to avoid modifying the original filters
    peptide_filters_to_apply = copy.deepcopy(filters_to_apply)

    # Function to flatten nested lists of filters
    def flatten_list(nested_list):
        """Flatten a nested list into a single list."""
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    # Remove percentile suffixes from filter column names for peptides
    for filter_dict in flatten_list(peptide_filters_to_apply):
        filter_dict['column'] = re.sub(r'_p\d{2,3}', '', filter_dict['column'])

    # Apply filters to the peptide metrics DataFrame
    peptide_summary_df, filtered_peptides_dfs = apply_filters_sequentially(
        peptide_metrics_df,
        filters=peptide_filters_to_apply,
        only_summary=False
    )
    # Extract the final filtered peptide DataFrame
    filtered_peptides_df = filtered_peptides_dfs[-1] # type: ignore

    return (
        gene_summary_df, 
        peptide_summary_df, 
        filtered_genes_df, 
        filtered_peptides_df
    ) # type: ignore
