import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any

UI_CONFIG = {        
        
    # Radio.
    'human_id_gene_rule': {
        'text': 'Rule for gene filtering',
        'options': ['Use mean values over gene peptides', 'Use minimum percentage of peptides'],
        'index': 1,
        'help': 'Select the rule to filter genes (peptide filtering is not affected by this).' 
    },
    
    # Slider.
    'human_id_gene_pc': {
        'text': 'Gene peptides (%) that need to pass',
        'default': 75,             
        'options': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50, 60, 70, 75, 78, 80, 82, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
        'help': 'Chooses which percentage of the gene peptides need to pass the filter for the gene to be retained.'
    },
    
    # Radio.
    'human_id_joint_rule': {
        'text': 'Filter combination (identity and length)',
        'options': [
            "Joint (use both conditions in same filter)", 
            "Independent (evaluated sequentially)"
        ],
        'index': 0,  # Default to "Joint", 
        'help': 'Select the way the filtering conditions are applied to the data for human identity filtering.'
    },
    
    # Slider.
    'identity_percent': {
        'text': 'Identity Percent (maximum)',
        'min_value': 0,
        'max_value': 100,
        'default': 80, 
        'step': 1, 
        'help': 'Identity percentage between the sequences matched.'
    },
    
    # Slider.
    'alignment_length': {
        'text': 'Alignment Length (maximum)',
        'min_value': 0,
        'max_value': 20,
        'default': 15, 
        'step': 1, 
        'help': 'Sets the alignment length (number of AAs) threshold for filtering.'
    },
    
    # Radio.
    'haplotype_gene_rule': {
        'text': 'Rule for gene filtering',
        'options': [
            "Use frequency of full exon haplotype",
            "Use minimum percentage of peptides"
        ],
        'index': 1,  # Default to first option
        'help': 'Select the rule to filter genes (peptide filtering is not affected by this).'
    },
    
    # Slider.
    'haplotype_gene_pc': {
        'text': 'Gene peptides (%) that need to pass',
        'options': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50, 60, 70, 75, 78, 80, 82, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
        'default': 95,
        'help': 'Chooses which percentage of the gene peptides need to pass the filter for the gene to be retained.'
    },
    
    # Radio.
    'haplotype_type': {
        'text': 'Haplotype Selection',
        'options': ["Identical to 3D7", "Any haplotype"],
        'index': 1, 
        'help': 'Select between the top haplotype required to be identical to 3D7 or to allow any haplotype.'
    },
    
    # Slider.
    'strain_conservation': {
        'text': 'AA Haplotype Frequency (minimum allowed)',
        'min_value': 0.0,
        'max_value': 1.0,
        'step': 0.01,
        'default': 0.99,
        'help': 'Frequency of the AA haplotype in field African samples.'
    },
    
    # Slider.
    'indel_frequency': {
        'text': 'Indel Frequency (maximum allowed)',
        'min_value': 0.0,
        'max_value': 1.0,
        'step': 0.01,
        'default': 0.05,
        'help': 'Max frequency of an indel allowed in a peptide.'
    },
    
    # Slider.
    'indel_gene_pc': {
        'text': 'Gene peptides (%) that need to pass',
        'options': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50, 60, 70, 75, 78, 80, 82, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
        'default': 100,
        'help': 'Chooses which percentage of the gene peptides need to pass the filter for the gene to be retained.'
    },
    
    # Radio.
    'indel_type': {
        'text': 'Indel Type',
        'options': ["Any", "Only Insertions", "Only Deletions"],
        'index': 0,  # Default to "Any"
        'help': 'Select the type of indel to consider (any, only insertions, only deletions).'
    },
    
    # Radio.
    'indel_frameshfits': {
        'text': 'Indel Effect',
        'options': ["Any", "Only Frameshifts"],
        'index': 0,  # Default to "Any"
        'help': 'Consider all indels or only those causing frameshifts.'
    },
    
    # Radio.
    'expression_rule': {
        'text': 'Rule',
        'options': ["At least one replicate", "At least two replicates", "All replicates"],
        'index': 2,
        'help': 'Select the rule to call the expression data (genes expressed).'
    },
    
    # Slider.
    'cpm_cutoff': {
        'text': 'CPM Cutoff (>=)',
        'min_value': 1,
        'max_value': 200,
        'step': 1,
        'default': 1,
        'help': 'Set the minimum CPM (Counts Per Million) cutoff value for calling expressed genes.'
    },
    
    # Multiselect.
    'expression_stage': {
        'text': 'Expression Stage',
        'options': [
            "Liver Stage Day 2",
            "Liver Stage Day 4",
            "Liver Stage Day 5",
            "Liver Stage Day 6",
            "Sporozoite Stage"
        ],
        'default': [],
        'help': 'Genes will be required to be expressed in the selected days/stages.'
    },
    
    # Multiselect.
    'homology_species': {
        'text': 'Homology in Species',
        'options': [
            "P. berghei",
            "P. vivax",
            "P. knowlesi"
        ],
        'default': [],
        'help': 'Genes will be required to have orthologs in the selected species.'
    }
}

def create_component(component_key: str, type: str, config: Dict[str, Dict[str, Any]]) -> None:
    """
    Create a Streamlit UI component and store its value in the session state.

    This function generates a specified Streamlit UI component (e.g., radio buttons, sliders,
    select sliders, multiselects) based on the provided configuration and stores the user's
    selection in `st.session_state` using the given component key.

    Parameters
    ----------
    component_key : str
        A unique identifier for the component. This key is used to access the component's
        configuration in the `config` dictionary and to store the component's value in
        `st.session_state`.

    type : str
        The type of Streamlit component to create. Supported types include:
        
        - `'radio'`: A set of radio buttons for selecting one option from a list.
        - `'select_slider'`: A slider with selectable values.
        - `'slider'`: A numeric slider for selecting a value within a range.
        - `'multiselect'`: A widget for selecting multiple options from a list.

    config : Dict[str, Dict[str, Any]]
        A nested dictionary containing configuration parameters for each component. The outer
        dictionary keys correspond to `component_key` values, and each inner dictionary contains
        the specific settings required for the component type.

        Example structure:
        
        ```python
        config = {
            'color_choice': {
                'text': 'Choose your favorite color:',
                'options': ['Red', 'Green', 'Blue'],
                'index': 1,  # For 'radio'
                'help': 'Select one color.'
            },
            'size_selection': {
                'text': 'Select size:',
                'options': ['Small', 'Medium', 'Large'],
                'default': 'Medium',  # For 'select_slider'
                'help': 'Choose the size that fits you best.'
            },
            'volume_adjust': {
                'text': 'Adjust volume:',
                'min_value': 0,          # For 'slider'
                'max_value': 100,
                'step': 5,
                'default': 50,
                'help': 'Use the slider to set the volume level.'
            },
            'hobbies_selection': {
                'text': 'Select your hobbies:',
                'options': ['Reading', 'Traveling', 'Gaming', 'Cooking'],
                'default': ['Reading', 'Gaming'],  # For 'multiselect'
                'help': 'Choose one or more hobbies.'
            }
            # Additional components can be added here...
        }
        ```

    Returns
    -------
    None
        The function does not return any value. Instead, it stores the selected value of the
        component in `st.session_state` under the provided `component_key`.

    Raises
    ------
    ValueError
        If an unsupported `type` is provided.

    Examples
    --------
    ```python
    import streamlit as st

    # Define the configuration for each component
    config = {
        'color_choice': {
            'text': 'Choose your favorite color:',
            'options': ['Red', 'Green', 'Blue'],
            'index': 1,
            'help': 'Select one color.'
        },
        'size_selection': {
            'text': 'Select size:',
            'options': ['Small', 'Medium', 'Large'],
            'default': 'Medium',
            'help': 'Choose the size that fits you best.'
        },
        'volume_adjust': {
            'text': 'Adjust volume:',
            'min_value': 0,
            'max_value': 100,
            'step': 5,
            'default': 50,
            'help': 'Use the slider to set the volume level.'
        },
        'hobbies_selection': {
            'text': 'Select your hobbies:',
            'options': ['Reading', 'Traveling', 'Gaming', 'Cooking'],
            'default': ['Reading', 'Gaming'],
            'help': 'Choose one or more hobbies.'
        }
    }

    # Create Streamlit components
    create_component('color_choice', 'radio', config)
    create_component('size_selection', 'select_slider', config)
    create_component('volume_adjust', 'slider', config)
    create_component('hobbies_selection', 'multiselect', config)

    # Accessing the selected values from session state
    st.write("Selected Color:", st.session_state.get('color_choice'))
    st.write("Selected Size:", st.session_state.get('size_selection'))
    st.write("Adjusted Volume:", st.session_state.get('volume_adjust'))
    st.write("Selected Hobbies:", st.session_state.get('hobbies_selection'))
    ```

    After running the above example, the selected values can be accessed via `st.session_state`:
    
    - `st.session_state['color_choice']` might return `'Green'`
    - `st.session_state['size_selection']` might return `'Medium'`
    - `st.session_state['volume_adjust']` might return `50`
    - `st.session_state['hobbies_selection']` might return `['Reading', 'Gaming']`

    These values can then be used elsewhere in your Streamlit application as needed.
    """
    if type == 'radio':       
        st.session_state[component_key] = st.radio(
            key=f'c{component_key}',
            label=config[component_key]['text'],
            options=config[component_key]['options'],
            index=config[component_key]['index'],
            help=config[component_key]['help']
        )
    elif type == 'select_slider':
        st.session_state[component_key] = st.select_slider(
            key=f'c{component_key}',
            label=config[component_key]['text'],
            options=config[component_key]['options'],
            value=config[component_key]['default'],
            help=config[component_key]['help']
        )
    elif type == 'slider':
        st.session_state[component_key] = st.slider(
            key=f'c{component_key}',
            label=config[component_key]['text'],
            min_value=config[component_key]['min_value'],
            max_value=config[component_key]['max_value'],
            step=config[component_key]['step'],
            value=config[component_key]['default'],
            help=config[component_key]['help']
        )
    elif type == 'multiselect':
        st.session_state[component_key] = st.multiselect(
            key=f'c{component_key}',
            label=config[component_key]['text'],
            options=config[component_key]['options'],
            default=config[component_key]['default'],
            help=config[component_key]['help']
        )
    else:
        raise ValueError(f'Unknown component type {type}')


def show_human_identity_filters(
    text_config: dict,
    config: dict,
):
    """
    Displays the human identity filters.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the filters.
    config : dict
        Configuration dictionary for UI components.
    """
    # Render the text for the filter.
    filter_text = text_config['filters']['human_identity_filter']
    st.write(filter_text['title'])
    for paragraph in filter_text['description']:
        st.write(paragraph)
    # Expander explanations.
    if filter_text.get('expander'):
        with st.expander(filter_text['expander']['title']):
            for section in filter_text['expander']['content']:
                if 'title' in section:
                    st.write(section['title'])
                for paragraph in section['paragraphs']:
                    st.write(paragraph)

    # Gene filtering rules: mean vs. minimum over peptides.
    col_rule, col_pct = st.columns([1.5,1])
    # Rule radio options (mean/minimum)
    with col_rule:
        create_component('human_id_gene_rule', 'radio', config)        
    # Slider with the percentage of gene peptides.
    with col_pct:
        create_component('human_id_gene_pc', 'select_slider', config)                
        
    # Choose filter combination display.
    create_component('human_id_joint_rule', 'radio', config)    
    
    pident_col, length_col = st.columns(2)                        
    # Identity Percent Slider
    with pident_col:
        create_component('identity_percent', 'slider', config)    
    # Alignment Length Slider
    with length_col:    
        create_component('alignment_length', 'slider', config)


def show_strain_conservation_filters(
    text_config: dict,
    config: dict
):
    """
    Displays strain conservation filters.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the filters.
    config : dict
        Configuration dictionary for UI components.
    """
    # Render the text for the filter.
    filter_text = text_config['filters']['strain_conservation_filter']
    st.write(filter_text['title'])
    for paragraph in filter_text['description']:
        st.write(paragraph)

    if filter_text.get('expander'):
        with st.expander(filter_text['expander']['title']):
            for section in filter_text['expander']['content']:
                if 'title' in section:
                    st.write(section['title'])
                for paragraph in section['paragraphs']:
                    st.write(paragraph)

    # Gene filtering rules.
    col1, col2 = st.columns([1.5, 1])
    # Rule radio options
    with col1:
        create_component('haplotype_gene_rule', 'radio', config)        
    with col2:
        create_component('haplotype_gene_pc', 'select_slider', config)        

    # Haplotype Selection Radio Button
    create_component('haplotype_type', 'radio', config)       
    # Strain Conservation Slider
    create_component('strain_conservation', 'slider', config)       
    
    
def show_indel_frequency_filters(
    text_config: dict,
    config: dict
):
    """
    Displays indel frequency filters.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the filters.
    config : dict
        Configuration dictionary for UI components.
    """
    
    # Render the text for the filter.
    filter_text = text_config['filters']['indel_frequency_filter']
    st.write(filter_text['title'])
    for paragraph in filter_text['description']:
        st.write(paragraph)

    if filter_text.get('expander'):
        with st.expander(filter_text['expander']['title']):
            for section in filter_text['expander']['content']:
                if 'title' in section:
                    st.write(section['title'])
                for paragraph in section['paragraphs']:
                    st.write(paragraph)

    # Options (left) and sliders (right).
    col1, col2 = st.columns([1.5, 1])
    with col1:
        # Type of indel.
        create_component('indel_type', 'radio', config)
        # Indel effect (frameshifts)
        create_component('indel_frameshfits', 'radio', config)        
    with col2:            
        # Max indel frequency.
        create_component('indel_frequency', 'slider', config)
        # Percentage of peptides per gene.
        create_component('indel_gene_pc', 'select_slider', config)
    
    
def show_gene_expression_filters(text_config: dict, config: dict):
    """
    Displays gene expression filters.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the filters.
    config : dict
        Configuration dictionary for UI components.
    """
    # Render the text for the filter.
    filter_text = text_config['filters']['gene_expression_filter']
    st.write(filter_text['title'])
    for paragraph in filter_text['description']:
        st.write(paragraph)

    if filter_text.get('expander'):
        with st.expander(filter_text['expander']['title']):
            for section in filter_text['expander']['content']:
                if 'title' in section:
                    st.write(section['title'])
                for paragraph in section['paragraphs']:
                    st.write(paragraph)

    # Create two columns for layout control
    col1, col2 = st.columns(2)
    # Rule radio options
    with col1:
        create_component('expression_rule', 'radio', config)    
    # CPM cutoff slider
    with col2:
        create_component('cpm_cutoff', 'slider', config)        

    # Expression Stages Multiselect
    create_component('expression_stage', 'multiselect', config)   

    
def show_gene_homology_filters(text_config: dict, config: dict):
    """
    Displays gene homology filters.

    Parameters
    ----------
    text_config : dict
        Dictionary containing text configurations for the filters.
    config : dict
        Configuration dictionary for UI components.
    """
    filter_text = text_config['filters']['gene_homology_filter']
    st.write(filter_text['title'])
    for paragraph in filter_text['description']:
        st.write(paragraph)

    # Homology Species Multiselect
    create_component('homology_species', 'multiselect', config)    
