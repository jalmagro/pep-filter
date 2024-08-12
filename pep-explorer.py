import gzip
import streamlit as st
import pandas as pd
import io

# Set of filters that can be applied to the datasets.
FILTERS = {
    
    "f_strain_conservation" : {
        'label': 'Strain Conservation', 
        'column': 'hap_3D7_freq',
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

def dataframe_to_csv(df, compress_data=False):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    if not compress_data:
        return csv_buffer.getvalue().encode('utf-8')
    else:
        gzipped_csv = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
        return gzipped_csv


def apply_filters_sequentially(df, filters, only_summary=False):
    """
    Apply a list of filters sequentially to a DataFrame and return the filtered DataFrames
    and a summary DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be filtered.
        filters (list of dict): A list of filter specifications.

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
    for i, filter_dict in enumerate(filters):
        label = filter_dict['label']
        column = filter_dict['column']
        op = filter_dict['op']
        value = filter_dict['value']
        
        # Create a filter description
        filter_description = f"{label} {op} {value}"
        
        # Apply the filter
        if op == '>=':
            filtered_df = current_df[current_df[column] >= value]
        elif op == '<=':
            filtered_df = current_df[current_df[column] <= value]
        elif op == '==':
            filtered_df = current_df[current_df[column] == value]
        else:
            raise ValueError(f"Unsupported filter operation: {op}")
        
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
    return summary_df, filtered_df

def call_expression_data(df, stage, cutoff, min_num_replicates, total_replicates):    
    return (df[[f'exp_{stage}_cpm_r{i+1}' for i in range(total_replicates)]] >= cutoff).sum(axis=1) >= min_num_replicates

def filter_datasets(
    gene_metrics_df, 
    pepetide_metrics_df, 
    identity_percent, 
    alignment_length,
    strain_conservation, 
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
    conservation_filter = FILTERS['f_strain_conservation'].copy()
    conservation_filter['value'] = strain_conservation
    
    # Compose the list of filters to apply.
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

    # Create summary dataframes (example)
    gene_summary_df, filtered_genes_df = apply_filters_sequentially(gene_metrics_df, filters_to_apply, only_summary=False)
    peptide_summary_df, filtered_peptides_df = apply_filters_sequentially(pepetide_metrics_df, filters_to_apply, only_summary=False)

    return gene_summary_df, peptide_summary_df, filtered_genes_df, filtered_peptides_df


def show_human_identity_filters():
                
    st.write("#### Human Identity Filtering") 
    st.write("""The sliders below allow you to filter BLAST results based on the percentage identity
            and alignment length of the Pf peptide sequences. The first slider controls the maximum percentage identity
            allowed, which is the proportion of AAs that match exactly between the Pf peptide an human exon sequence. 
            The alignment length slider determines the maximum allowed length of the aligned region between the two sequences.""")

    with st.expander("Show me an example and more info"):
        st.write("""**Example**: Suppose you set the identity slider to 90%. This means that only alignments where 90% or less 
                    of the amino acids match between the sequences will be shown. If you set the alignment 
                    length slider to 18, only alignments that cover 18 amino acids or less of the 20-AA peptides will 
                    be considered. This combination would filter out any alignments where the percentage identity is 
                    more than 90% or the alignment covers more than 18 amino acids.""")
        st.write("""Notice this filter is designed to work with peptides, values for genes are extrapolated by considering
                    all the peptides in a gene and computing the mean of each metric (which dilutes the signal since most peptides
                    have no matches).                        
                    """)

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
    st.write("""This slider determines the minimum frequency of the 3D7 AA haplotype observed in all Pf7 African samples (n ~ 8000)
                for a candidate peptide or gene to be considered.""")
    with st.expander("Show me an example and more info"):
        st.write("""**Example**: Suppose you set the frequency slider to 1.0, this means that for a candidate peptide or gene exon to 
                    be retained the only haplotype observed in African field samples must be 3D7 (at a frequency of 100%).
                    We recommend a less stringent filtering value (e.g., 0.99 or 0.95), as this would accomodate sequencing errors and 
                    sporadic rare variants.""")                      
        st.write("""Notice this filter behaves differently in genes and peptides. The longer the haplotype, the more likely it is 
                    to accumulate mutations, which means that many peptide haplotypes (20-AAs) have no mutations at all 
                    (a single fixed haplotype at 100% frequency) whereas long genes are more likely to stratify into different haplotypes.
                    """)

    
    st.session_state.strain_conservation = st.slider(
        "3D7 AA Haplotype Frequency (minimum allowed)",
        min_value=0.0, max_value=1.0, step=0.01, value=st.session_state.strain_conservation,
        help="""Frequency of the 3D7 haplotype in field African samples."""
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
        st.image("logo-med.png", use_column_width=True)
    
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
        
    # Init initial summary stats for filtering.
    if 'summary_genes_df' not in st.session_state:
    # Filter the datasets and store them in session state
        summary_genes_df, summary_peptides_df, filtered_genes_df, filtered_peptides_df = filter_datasets(
            st.session_state.gene_metrics_df,
            st.session_state.peptide_metrics_df,
            st.session_state.identity_percent,
            st.session_state.alignment_length,
            st.session_state.strain_conservation,
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
                st.session_state.identity_percent,
                st.session_state.alignment_length,
                st.session_state.strain_conservation,
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

    # Display summary statistics
    gene_col, peptide_col = st.columns([1, 1])  # Adjust the ratio to control the width    
    with gene_col:
        st.write("### Gene Filtering")
        st.dataframe(st.session_state.summary_genes_df.set_index('filter').drop('order', axis=1))
    with peptide_col:
        st.write("### Peptide Filtering")
        st.dataframe(st.session_state.summary_peptides_df.set_index('filter').drop('order', axis=1))


    gene_col_download, peptide_col_download = st.columns([1, 1])
    with gene_col_download:
        st.download_button(
            label="Download List of Candidate Genes",
            data=dataframe_to_csv(st.session_state.filtered_genes_df[['gene_id', 'chromosome', 'start', 'end', 'gene_name', 'plasmo_db_url', 'num_peptides']]),
            file_name="candidate-genes.csv",
            mime="text/csv"
        )
    with peptide_col_download:
        peptide_results_df = st.session_state.filtered_peptides_df[['peptide_id', 'gene_id', 'start', 'end', 'peptide', 'chromosome', 'gene_name', 'plasmo_db_url']]
        compress_data = False
        if len(peptide_results_df) > 20000:
            compress_data = True
        st.download_button(
            label="Download List of Candidate Peptides",
            data=dataframe_to_csv(peptide_results_df, compress_data),
            file_name="candidate-peptides.csv" + ".gz" if compress_data else '',
            mime="text/csv" if not compress_data else "application/gzip" 
        )

# Main navigation
def main():
    
    # Load the original datasets into session state (only once).
    # These are big files, it will take a few seconds.
    if 'gene_metrics_df' not in st.session_state:
        st.session_state.gene_metrics_df = pd.read_csv('data/gene-metrics-filtering.csv.gz')  
    if 'peptide_metrics_df' not in st.session_state:
        st.session_state.peptide_metrics_df = pd.read_csv('data/peptide-metrics-filtering.csv.gz')  

    show_filters()
    
if __name__ == "__main__":
    main()
