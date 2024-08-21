import gzip
import re
import streamlit as st
import pandas as pd
import numpy as np
import io
import itertools

# Set of filters that can be applied to the datasets.
FILTERS = {
    
    "f_strain_conservation_3D7" : {
        'label': 'Strain Conservation', 
        'column': 'hap_3D7_freq',
        'op': '>=',
        'value': 0.99
    },
    
    "f_strain_conservation_any_hap" : {
        'label': 'Strain Conservation', 
        'column': 'hap_top_hap_freq',
        'op': '>=',
        'value': 0.99
    },
    
    "f_human_identity_filter" : {
        'label': 'Human Identity %', 
        'column': 'blt_pident',
        'op': '<=',
        'value': 60
    },
    
    "f_human_length_filter" : {
        'label': 'Human Alignment Lenght', 
        'column': 'blt_length',
        'op': '<=',
        'value': 15
    },

    "f_pb_homology" : {
        'label': 'Homology in P. berghei', 
        'column': 'hom_homolog_in_pb',
        'op': '==',
        'value': True
    },

    "f_pv_homology" : {
        'label': 'Homology in P. vivax',
        'column': 'hom_homolog_in_pv', 
        'op': '==',
        'value': True
    },
    
    "f_pk_homology" : {
        'label': 'Homology in P. knowlesi',
        'column': 'hom_homolog_in_pk', 
        'op': '==',
        'value': True
    },

    "f_liver_expression_d2" : {
        'label': 'Liver-expressed D2', 
        'op': '==',
        'column': 'exp_expressed_d2', 
        'value': True
    },

    "f_liver_expression_d4" : {
        'label': 'Liver-expressed D4', 
        'op': '==',
        'column': 'exp_expressed_d4', 
        'value': True
    },
    
    "f_liver_expression_d5" : {  
        'label': 'Liver-expressed D5', 
        'op': '==',
        'column': 'exp_expressed_d5', 
        'value': True
    },

    "f_liver_expression_d6" : {  
        'label': 'Liver-expressed D6', 
        'op': '==',
        'column': 'exp_expressed_d6', 
        'value': True
    },

    "f_liver_expression_sporozoite" : {
        'label': 'Expressed in Sporozoite', 
        'op': '==',
        'column': 'exp_expressed_sporozoite', 
        'value': True
    }
}

def flatten_list(nested_list):
    """Flatten a list that may contain nested lists and single elements."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def dataframe_to_csv(df, compress_data=False):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    if not compress_data:
        return csv_buffer.getvalue().encode('utf-8')
    else:
        gzipped_csv = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
        return gzipped_csv

def render_dataframe_as_html(df):
    """Render the DataFrame as HTML with line breaks correctly displayed."""
    df = df.applymap(lambda x: str(x).replace('\n', '<br>') if isinstance(x, str) else x)
    df_html =df.to_html(escape=False)
    st.markdown(df_html, unsafe_allow_html=True)

def apply_filters_sequentially(df, filters, only_summary=False):
    """
    Apply a list of filters (which can include nested filters) sequentially to a DataFrame 
    and return the filtered DataFrames and a summary DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be filtered.
        filters (list of dict or list of list of dict): A list of filter specifications or nested lists of filter specifications.
        only_summary (bool): If True, only return the summary DataFrame.

    Returns:
        filtered_dfs (list of pd.DataFrame): A list of DataFrames resulting from applying each filter.
        summary_df (pd.DataFrame): A DataFrame containing the filter number, label, and row count after applying each filter.
    """
    filtered_dfs = []
    summary_data = []
    summary_data.append({
        'order': 0,
        'filter': 'Initial Dataset',
        'remaining': len(df),
        'fraction': 1.0
    })

    current_df = df.copy()
    
    for i, filter_group in enumerate(filters):
        if isinstance(filter_group, dict):
            # If it's a single filter (not nested), convert it to a list for consistency
            filter_group = [filter_group]

        combined_filter_description = []
        combined_condition = np.full(len(current_df), True)
        
        for filter_dict in filter_group:
            label = filter_dict['label']
            column = filter_dict['column']
            op = filter_dict['op']
            value = filter_dict['value']

            # Create a filter condition based on the operation
            if op == '>=':
                condition = current_df[column] >= value
            elif op == '<=':
                condition = current_df[column] <= value
            elif op == '==':
                condition = current_df[column] == value
            else:
                raise ValueError(f"Unsupported filter operation: {op}")
            
            # Combine conditions with logical AND
            combined_condition &= condition
            
            # Create a description for this individual filter
            combined_filter_description.append(f"{label} {op} {value}")
        
        # Join all individual filter descriptions with 'AND'
        filter_description = ' \n AND '.join(combined_filter_description)
        
        # Apply the combined filter to the DataFrame
        filtered_df = current_df[combined_condition]
        
        # Store the filtered DataFrame
        filtered_dfs.append(filtered_df)
        
        # Update the current DataFrame to the filtered result
        current_df = filtered_df.copy()
        
        # Store summary information
        summary_data.append({
            'order': i+1,
            'filter': filter_description,
            'remaining': len(filtered_df),
            'fraction': len(filtered_df)/len(df)
        })

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    if only_summary:
        return summary_df
    return summary_df, filtered_dfs


def call_expression_data(df, stage, cutoff, min_num_replicates, total_replicates):    
    return (df[[f'exp_{stage}_cpm_r{i+1}' for i in range(total_replicates)]] >= cutoff).sum(axis=1) >= min_num_replicates

def filter_datasets(
    gene_metrics_df, 
    pepetide_metrics_df,
    human_id_gene_rule,
    human_id_gene_pc,
    human_id_joint_rule,
    identity_percent, 
    alignment_length,
    strain_conservation,
    haplotype_gene_rule,
    haplotype_gene_pc,
    haplotype_type,
    expression_rule,
    cpm_cutoff,
    expression_stage, 
    homology_species
):        
    # Filters that are always executed.
    # Update their filtering values.
    human_id_filter = FILTERS['f_human_identity_filter'].copy()
    human_id_filter['value'] = identity_percent
    human_length_filter = FILTERS['f_human_length_filter'].copy()
    human_length_filter['value'] = alignment_length 
    
    # Get the right percentile column if required.
    if human_id_gene_rule == 'Use minimum percentage of peptides':
        human_id_filter['column'] += f'_p{human_id_gene_pc:02}'
        human_length_filter['column'] += f'_p{human_id_gene_pc:02}'

    # Choose the type of AA haplotype to filter for.
    if haplotype_type == 'Identical to 3D7':        
        conservation_filter = FILTERS['f_strain_conservation_3D7'].copy()                        
    else:        
        conservation_filter = FILTERS['f_strain_conservation_any_hap'].copy()    
    
    # Get the right percentile column if required.
    if haplotype_gene_rule == 'Use minimum percentage of peptides':
        conservation_filter['column'] += f'_p{(100 - haplotype_gene_pc):02}'        
    
    conservation_filter['value'] = strain_conservation
    
    # Compose the list of filters to apply.
    # Check if we need to nest filters (so the conditions are computed simultaneously).
    if human_id_joint_rule == "Joint (use both conditions in same filter)":
        filters_to_apply = [[human_id_filter, human_length_filter], conservation_filter]
    else:
        filters_to_apply = [human_id_filter, human_length_filter, conservation_filter]
            
    # Recall gene expression.
    if expression_rule == "At least one replicate":
        min_num_replicates = 1
    elif expression_rule == "At least two replicates":
        min_num_replicates = 2
    elif expression_rule == "All replicates":
        min_num_replicates = 3
    else:
        raise ValueError(f'Unknown Rule {expression_rule}')
    
    # Liver Stages (days).
    gene_metrics_df['exp_expressed_d2'] = call_expression_data(gene_metrics_df, 'd2', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    gene_metrics_df['exp_expressed_d4'] = call_expression_data(gene_metrics_df, 'd4', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    gene_metrics_df['exp_expressed_d5'] = call_expression_data(gene_metrics_df, 'd5', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    gene_metrics_df['exp_expressed_d6'] = call_expression_data(gene_metrics_df, 'd6', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    
    pepetide_metrics_df['exp_expressed_d2'] = call_expression_data(pepetide_metrics_df, 'd2', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    pepetide_metrics_df['exp_expressed_d4'] = call_expression_data(pepetide_metrics_df, 'd4', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    pepetide_metrics_df['exp_expressed_d5'] = call_expression_data(pepetide_metrics_df, 'd5', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    pepetide_metrics_df['exp_expressed_d6'] = call_expression_data(pepetide_metrics_df, 'd6', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=3)
    
    # Sporozoite stage.
    if expression_rule == "All replicates":
        min_num_replicates = 5
    gene_metrics_df['exp_expressed_sporozoite'] = call_expression_data(gene_metrics_df, 'sporozoite', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=5)
    pepetide_metrics_df['exp_expressed_sporozoite'] = call_expression_data(pepetide_metrics_df, 'sporozoite', cutoff=cpm_cutoff, min_num_replicates=min_num_replicates, total_replicates=5)

    # Gene Expression.
    if 'Liver Stage Day 2' in expression_stage:
        filters_to_apply.append(FILTERS['f_liver_expression_d2'])
    if 'Liver Stage Day 4' in expression_stage:
        filters_to_apply.append(FILTERS['f_liver_expression_d4'])
    if 'Liver Stage Day 5' in expression_stage:
        filters_to_apply.append(FILTERS['f_liver_expression_d5'])
    if 'Liver Stage Day 6' in expression_stage:
        filters_to_apply.append(FILTERS['f_liver_expression_d6'])
    if 'Sporozoite Stage' in expression_stage:
        filters_to_apply.append(FILTERS['f_liver_expression_sporozoite'])
        
    # Gene Homology.
    if 'P. berghei' in homology_species:
        filters_to_apply.append(FILTERS['f_pb_homology'])
    if 'P. vivax' in homology_species:
        filters_to_apply.append(FILTERS['f_pv_homology'])
    if 'P. knowlesi' in homology_species:
        filters_to_apply.append(FILTERS['f_pk_homology'])

    # Create summary dataframes for genes.    
    gene_summary_df, filtered_genes_df = apply_filters_sequentially(gene_metrics_df, filters_to_apply, only_summary=False)
    # Retain only the final filtering results.
    filtered_genes_df = filtered_genes_df[-1]
    
    # And peptides.
    # HACK: We need to create separate filter lists for gene and peptides as now it is possible for the filters to target
    #  different columns (e.g., percentiles). For the time being we just replace the _pXX suffixes to fix the problem
    # Notice we need to flatten the list to deal with nested filters.
    for f in flatten_list(filters_to_apply) :
        f['column'] = re.sub(r'_p\d{2,3}', '', f['column'])
    peptide_summary_df, filtered_peptides_df = apply_filters_sequentially(pepetide_metrics_df, filters_to_apply, only_summary=False)
    # Retain only the final filtering results.
    filtered_peptides_df = filtered_peptides_df[-1]

    return gene_summary_df, peptide_summary_df, filtered_genes_df, filtered_peptides_df


def show_human_identity_filters():
                
    st.write("#### Human Identity Filtering") 
    st.write("""The sliders below allow you to filter BLAST results based on the percentage identity
            and alignment length of the Pf peptide sequences. The first slider controls the maximum percentage identity
            allowed, which is the proportion of AAs that match exactly between the Pf peptide an human exon sequence. 
            The alignment length slider determines the maximum allowed length of the aligned region between the two sequences.""")

    with st.expander("Show me an example and more details about the filtering process"):
        
        st.write("""#### Using the Filters""")
        st.write(""" - **Identity Slider**: If you set the identity slider to 90%, only alignments where 90% or fewer of the amino acids match between sequences will be shown.""")
        st.write(""" - **Alignment Length Slider**: If you set the alignment length slider to 18, only alignments covering 18 or fewer amino acids of the 20-AA peptides will be considered.""")
            
        st.write("""#### Filter Combination""")
        st.write("""If you want both conditions (identity ≤ 90% AND alignment length ≤ 18) to be be shown simultaneously, select "Joint" as the Filter Combination option. Otherwise, the filters will be executed sequentially. **Notice you'll obtain the same results** it only changes the way conditions are groupe into filters.""")
    
        st.write("""#### Gene Filtering Options""")
        st.write("""This filter is primarily designed to work with peptides, but you can choose how values for genes are extrapolated using the **Rule for Gene Filtering** options:""")
        st.write("""##### Use Mean Values Over Gene Peptides""")
        st.write("""This option calculates the average of the metrics across all peptides in a gene. While this approach may dilute the signal (since many peptides may not have matches), it provides an overall metric for the gene.""")
        st.write("""##### Use Minimum Percentage of Gene Peptides""")
        st.write("""Alternatively, you can filter out a gene if a percentage of its peptides don't pass both filters.""")
        st.write("""#### Example: Filtering by Percentage of Peptides""")
        st.write("""For example, if you set the alignment length slider to 18 amino acids (AAs) and decide to use 
                    a minimum peptide percentage of 90%, you will filter out any gene 
                    for which less than 90% of their peptides have a frequency above 0.99.""")
        st.write("""#### Important Note""")
        st.write("""Peptides are always evaluated individually. The gene extrapolation rules apply only to the way gene metrics are calculated, not to individual peptides.""")

    # Gene filtering rules.
    col1, col2 = st.columns([1.5,1])
    # Rule radio options
    with col1:
        st.session_state.human_id_gene_rule = st.radio(
            "Rule for gene filtering",
            options=["Use mean values over gene peptides", "Use minimum percentage of peptides"],
            index=0, 
            help="Select the rule to filter genes (peptide filtering is not affected by this)."
        )
    # CPM cutoff slider
    with col2:        
        st.session_state.human_id_gene_pc = st.select_slider(
            "Percentage of gene peptides",
            options=[0,1,5,10,25,50,75,90,95,99,100],
            value=st.session_state.human_id_gene_pc, 
            help="Chooses which percentage of the gene peptides need to pass the filter for the gene to be retained."        
        )
    
    # # Choose filter mode    
    st.session_state.human_id_joint_rule = st.radio(
        "Filter combination (identity and length)",
        options=["Joint (use both conditions in same filter)", "Independent (evaluated sequentially)"],
        index=0,  # Default to "All replicates"
        help="Select the way the filtering conditions are applied to the data for human identity filtering."
    )
    
    pident_col, length_col = st.columns(2)                    
    # Identity Percent Slider
    with pident_col:
        st.session_state.identity_percent = st.slider(
            "Identity Percent (maximum)",
            min_value=0, max_value=100, value=st.session_state.identity_percent,  
            help="Identity percentage between the sequences matched."
        )        
    # Alignment Length Slider
    with length_col:
        st.session_state.alignment_length = st.slider(
            "Alignment Length (maximum)",
            min_value=0, max_value=20, value=st.session_state.alignment_length,  
            help="Sets the alignment length (number of AAs) threshold for filtering."
        )
        
        
def show_strain_conservation_filters():
    
    st.write("#### Strain Conservation")
    st.write("""This slider determines the minimum frequency of the AA haplotype observed in all Pf7 African samples (n ~ 8000)
                for a candidate peptide or gene to be considered.""")
    with st.expander("Show me an example and more info"):
        
        st.write("""#### Example: Stringent Frequency""")
        st.write("""Suppose you set the frequency slider to 1.0, this means that for a candidate peptide or gene to 
                    be retained the only haplotype observed in African field samples must be at a frequency of 100%.
                    We recommend a less stringent filtering value (e.g., 0.99 or 0.95), as this would accomodate sequencing errors and 
                    sporadic rare variants.""")  
        
        st.write("""#### Haplotype Selection""")
        st.write("""When filtering by frequency, you can choose between restricting the filtering to haplotypes that are identical to 3D7 
                    (the reference genome) or to consider any haplotype independently of its mutations.""")  
        
        st.write("""#### Gene Filtering Options""")
        st.write("""You can choose how values for genes are computed using the **Rule for Gene Filtering** options:""")
        st.write("""##### Use Frequency of Full Gene Haplotype""")
        st.write("""This option uses a single haplotype covering the whole gene (exons) to represent the gene.""")
        st.write("""##### Use Minimum Percentage of Gene Peptides""")
        st.write("""Alternatively, you can filter out a gene if a percentage of its peptides don't pass the frequency filter.""")
        st.write("""#### Example: Filtering by Percentage of Peptides""")
        st.write("""For example, if you set the minimum allowed frequency to 0.99 and decide to use 
                    a minimum peptide percentage of 90%, you will filter out any gene 
                    for which less than 90% of their peptides have a frequency above 0.99.""")
    
        st.write("""#### Important Note""")
        st.write("""Peptides are always evaluated individually. The gene extrapolation rules apply only to the way gene metrics are calculated, not to individual peptides.""")

    
    # Gene filtering rules.
    col1, col2 = st.columns([1.5,1])
    # Rule radio options
    with col1:
        st.session_state.haplotype_gene_rule = st.radio(
            "Rule for gene filtering",
            options=["Use frequency of full exon haplotype", "Use minimum percentage of peptides"],
            index=0, 
            help="Select the rule to filter genes (peptide filtering is not affected by this)."
        )
    # CPM cutoff slider
    with col2:        
        st.session_state.haplotype_gene_pc = st.select_slider(
            "Percentage of gene peptides",
            options=[0,1,5,10,25,50,75,90,95,99,100],
            value=st.session_state.haplotype_gene_pc, 
            help="Chooses which percentage of the gene peptides need to pass the filter for the gene to be retained."        
        )

    st.session_state.haplotype_type = st.radio(
        "Haplotype Selection",
        options=["Identical to 3D7", "Any haplotype"],
        help="Select between the top haplotype required to be identical 3D7 or to allow any haplotype."
    )

    st.session_state.strain_conservation = st.slider(
        "AA Haplotype Frequency (minimum allowed)",
        min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.strain_conservation,
        help="""Frequency of the AA haplotype in field African samples."""
        )


def show_gene_expression_filters():
    
    st.write("#### Gene Expression")
    st.write("""In this section you can specify how expressed genes are detected and filter candidates by their expression status in
            the liver stage (days 2,4,5 and 6) or the sporozoite stage.""")
    
    with st.expander("Show me an example and more info"):
        
        st.write("""The dataset from Zhangi *et al*., contains three biological replicates for each liver stage day and five replicates
                    for the sporozoite stage. To decide if a gene has been expressed they follow the rule "using a cutoff of >= 1 CPM in 
                    the three biological replicates" but that seems inconsistent with some of the results presented in their paper
                    (we are awaiting for their response). To overcome this issue, here you can call which genes are expressed by choosing
                    the calling rule (e.g., "all replicates" or "at least one replicate"), and also specifying the minimum CPM cutoff.""")
        
        st.write("""Later on you can select in which days/stages a gene needs to be expressed in order to be considered a candidate. 
                    Notice the gene will have to be expressed in ALL days/stages selected. A peptide is considered expressed if its associated
                    gene is expressed.""")
        
        st.write("""**Example**: Suppose you set the calling rule to "at least one replicate" and the cutoff to 5 CPM. Later you select
                    Liver Stage Day 6. This means that only genes with at least one liver stage biological replicate for day 6 with 5 or more 
                    CPM will be retained.""") 

    # Create two columns for layout control
    col1, col2 = st.columns(2)

    # Rule radio options
    with col1:
        st.session_state.expression_rule = st.radio(
            "Rule",
            options=["At least one replicate", "At least two replicates", "All replicates"],
            index=2,  # Default to "All replicates"
            help="Select the rule to call the expression data (genes expressed)."
        )
    # CPM cutoff slider
    with col2:
        st.session_state.cpm_cutoff = st.slider(
            "CPM Cutoff (>=)",
            min_value=1, max_value=200, step=1, value=st.session_state.cpm_cutoff,  # Default value set to 10
            help="Set the minimum CPM (Counts Per Million) cutoff value for calling expressed genes."
        )
                    
    # Expression Checkboxes
    expression_options = [
        "Liver Stage Day 2",
        "Liver Stage Day 4",
        "Liver Stage Day 5",
        "Liver Stage Day 6",
        "Sporozoite Stage"
    ]
    st.session_state.expression_stage = st.multiselect(
        "Expression Stage",
        expression_options, default=st.session_state.expression_stage,
        help="""Genes will be required to be expressed in the selected days/stages."""
    )


def show_gene_homology_filters():
    st.write("#### Gene Homology")
    st.write("""Genes will be required to be expressed in the selected days/stages.""")
    
    # Homology Checkboxes
    homology_options = [
        "P. berghei",
        "P. vivax",
        "P. knowlesi"
    ]
    st.session_state.homology_species = st.multiselect(
        "Homology in Species",
        homology_options, default=st.session_state.homology_species,
        help="""Genes will be required to have orthologs in the selected species."""
    )



def show_filters():
    
    col1, col2, col3 = st.columns([1, 4, 1])  # Adjust the column width ratios as needed
    with col2:
        st.image("logo-medium.png", use_column_width=True)
    
    st.title("PepExplorer Filtering Module")
    st.write("""This is a prototype of the filering module for the `PepExplorer` project. It allows the user to filter candidate genes/peptides 
             for potential vaccine targets using a set of filters. 
             Notice you can download the filtering summary by hovering over the results tables. To download the list of candidate genes/peptides
             use the buttons just below the summaries (note: for results with more than 20K peptides, the output file is compressed with gzip).""")
    st.write("""Genes and peptides are filtered independently (at the gene and peptide level) but some metrics are extrapolated 
             (see details for each filter below). **This is work in progress, and not yet ready for wider public use**. For feedback please contact `jg10@sanger.ac.uk`.
             """)

    # Initialize session state variables if they don't exist
    if 'identity_percent' not in st.session_state:
        st.session_state.identity_percent = 80  # Default value
    if 'alignment_length' not in st.session_state:
        st.session_state.alignment_length = 15  # Default value        
    if 'strain_conservation' not in st.session_state:
        st.session_state.strain_conservation = 0.99  # Default value
    if 'expression_rule' not in st.session_state:
        st.session_state.expression_rule = 'All replicates'
    if 'cpm_cutoff' not in st.session_state:
        st.session_state.cpm_cutoff = 1
    if 'expression_stage' not in st.session_state:
        st.session_state.expression_stage = []
    if 'homology_species' not in st.session_state:
        st.session_state.homology_species = []
    if 'haplotype_type' not in st.session_state:
        st.session_state.haplotype_type = "Identical to 3D7"
    if 'haplotype_gene_rule' not in st.session_state:
        st.session_state.haplotype_gene_rule = "Use frequency of full exon haplotype"
    if 'haplotype_gene_pc' not in st.session_state:
        st.session_state.haplotype_gene_pc = 95    
    if 'human_id_gene_rule' not in st.session_state:
        st.session_state.human_id_gene_rule = "Use mean values over gene peptides"
    if 'human_id_gene_pc' not in st.session_state:
        st.session_state.human_id_gene_pc = 75
    if 'human_id_joint_rule' not in st.session_state:
        st.session_state.human_id_joint_rule = "Joint (use both conditions in same filter)"      
        
    # Init initial summary stats for filtering.
    if 'summary_genes_df' not in st.session_state:
        # Filter the datasets and store them in session state
        summary_genes_df, summary_peptides_df, filtered_genes_df, filtered_peptides_df = filter_datasets(
            st.session_state.gene_metrics_df,
            st.session_state.peptide_metrics_df,
            st.session_state.human_id_gene_rule,
            st.session_state.human_id_gene_pc,
            st.session_state.human_id_joint_rule,            
            st.session_state.identity_percent,
            st.session_state.alignment_length,
            st.session_state.strain_conservation,
            st.session_state.haplotype_gene_rule,
            st.session_state.haplotype_gene_pc,
            st.session_state.haplotype_type,
            st.session_state.expression_rule,
            st.session_state.cpm_cutoff,
            st.session_state.expression_stage,
            st.session_state.homology_species,            
        )            
        # Save in the global state.
        st.session_state.summary_genes_df = summary_genes_df
        st.session_state.summary_peptides_df = summary_peptides_df
        st.session_state.filtered_genes_df = filtered_genes_df
        st.session_state.filtered_peptides_df = filtered_peptides_df

    # Form for filtering candidates
    with st.form(key='filter_form'):

        show_human_identity_filters()
        show_strain_conservation_filters()
        show_gene_expression_filters()
        show_gene_homology_filters()
                        
        st.write("")
        st.write("")
                
        # Submit button
        submit_button = st.form_submit_button(label='Filter Candidates')

        if submit_button:                                    
            # Filter the datasets and store them in session state
            summary_genes_df, summary_peptides_df, filtered_genes_df, filtered_peptides_df = filter_datasets(
                st.session_state.gene_metrics_df,
                st.session_state.peptide_metrics_df,
                st.session_state.human_id_gene_rule,
                st.session_state.human_id_gene_pc,
                st.session_state.human_id_joint_rule,
                st.session_state.identity_percent,
                st.session_state.alignment_length,
                st.session_state.strain_conservation,
                st.session_state.haplotype_gene_rule,
                st.session_state.haplotype_gene_pc,
                st.session_state.haplotype_type,
                st.session_state.expression_rule,
                st.session_state.cpm_cutoff,
                st.session_state.expression_stage,
                st.session_state.homology_species
            )            
            # Save in the global state.
            st.session_state.summary_genes_df = summary_genes_df
            st.session_state.summary_peptides_df = summary_peptides_df
            st.session_state.filtered_genes_df = filtered_genes_df
            st.session_state.filtered_peptides_df = filtered_peptides_df

    # Gene results.
    num_genes_retained = len(st.session_state.filtered_genes_df)
    num_total_peptides = st.session_state.filtered_genes_df.num_peptides.sum()    
    with st.container():
        st.write('\n')
        st.write("## Gene Filtering Results")
        st.write("Below is a summary of the filtering results. You have the option to download either the summary of the applied filters or the final list of filtered genes. Please note that genes are filtered independently from peptides, based on the filtering settings you’ve configured.")
        st.write("")
        
        st.dataframe(st.session_state.summary_genes_df.set_index('filter').drop('order', axis=1))    
        
        if st.session_state.human_id_gene_rule == "Use mean values over gene peptides":
            st.markdown(f'###### Human identity gene-extrapolation rule: `Mean over gene peptides`')
        else:            
            human_id_gene_pc = int(str(st.session_state.human_id_gene_pc).replace('min','0').replace('max','100'))            
            st.markdown(f'###### Human identity gene-extrapolation rule: `Minimum percentage of gene peptides`')
            st.write(f' - Up to **{100 - human_id_gene_pc}% of gene peptides can fail** identity and length filters')            
            
            
        if st.session_state.haplotype_gene_rule == "Use frequency of full exon haplotype":
            st.markdown(f'###### Strain conservation gene-extrapolation rule: `Full gene haplotype`')
        else:
            haplotype_gene_pc = int(str(st.session_state.haplotype_gene_pc).replace('min','0').replace('max','100'))
            st.markdown(f'###### Strain conservation gene-extrapolation rule: `Minimum percentage of gene peptides`')
            st.write(f'- At least **{haplotype_gene_pc}% of gene peptides must pass** frequency filter')
    
        st.write("\n")
        st.markdown(f'##### Genes retained: `{num_genes_retained:,}`')
        st.write(f'##### Total number of peptides: `{num_total_peptides:,}`')
        st.write(f'##### Average number of peptides per gene: `{round(num_total_peptides/num_genes_retained, 2):,}`')        
        st.write("")
        lc, rc, _ = st.columns([1,1,2])
        lc.download_button(
            label="Download Filtering Summary (Genes)",
            data=dataframe_to_csv(st.session_state.summary_genes_df),
            file_name="candidate-genes-filter-summary.csv",
            mime="text/csv"
        )
        rc.download_button(
            label="Download Table of Candidate Genes",
            data=dataframe_to_csv(st.session_state.filtered_genes_df[['gene_id', 'chromosome', 'start', 'end', 'gene_name', 'plasmo_db_url', 'num_peptides', 'hap_top_hap']]),
            file_name="candidate-genes.csv",
            mime="text/csv"
        )
    
    # Peptide results.    
    num_genes_retained = len(set(st.session_state.filtered_peptides_df.gene_id))
    num_total_peptides = len(st.session_state.filtered_peptides_df)
    with st.container():
        st.write('\n')
        st.write('\n')
        st.write("## Peptide Filtering Results")
        st.write("Below is a summary of the filtering results. You have the option to download either the summary of the applied filters or the final list of filtered peptides. Please note that genes are filtered independently from peptides, based on the filtering settings you’ve configured.")                
        st.dataframe(st.session_state.summary_peptides_df.set_index('filter').drop('order', axis=1))                
        st.write(f'##### Peptides retained: `{num_total_peptides:,}`')
        st.write(f'##### Genes involved (at least one peptide retained): `{num_genes_retained:,}`')
        st.write(f'##### Average number of peptides per gene: `{round(num_total_peptides/num_genes_retained, 2):,}`')        
        st.write("")
        
        peptide_results_df = st.session_state.filtered_peptides_df[['peptide_id', 'gene_id', 'start', 'end', 'peptide', 'chromosome', 'gene_name', 'plasmo_db_url', 'hap_top_hap']]
        compress_data = False
        if len(peptide_results_df) > 20000:
            compress_data = True
        
        def generate_csv_and_download(ctx):
            csv_data = dataframe_to_csv(peptide_results_df, compress_data)
            ctx.download_button(
                label="Download Table of Candidate Peptides",
                data=csv_data,
                file_name="candidate-peptides.csv" + (".gz" if compress_data else ''),
                mime="text/csv" if not compress_data else "application/gzip"
            )
            
        lc, rc, dc, _ = st.columns([1,1,1,1])
        lc.download_button(
            label="Download Filtering Summary (Peptides)",
            data=dataframe_to_csv(st.session_state.summary_peptides_df),
            file_name="candidate-genes-filter-summary.csv",
            mime="text/csv"
        )
        
        if rc.button("Prepare Download for Peptides"):
            generate_csv_and_download(dc)
        
        # rc.download_button(
        #     label="Download Table of Candidate Peptides",
        #     data=dataframe_to_csv(peptide_results_df, compress_data),
        #     file_name="candidate-peptides.csv" + ".gz" if compress_data else '',
        #     mime="text/csv" if not compress_data else "application/gzip" 
        # )

# Main navigation
def main():
    
    # Load the original datasets into session state (only once).
    # These are big files, it will take a few seconds.
    if 'gene_metrics_df' not in st.session_state:
        st.session_state.gene_metrics_df = pd.read_csv('data/gene-metrics-filtering.csv.gz')  
    if 'peptide_metrics_df' not in st.session_state:
        st.session_state.peptide_metrics_df = pd.read_csv('data/peptide-metrics-filtering.csv.gz', low_memory=False)  

    show_filters()
    
if __name__ == "__main__":
    main()
