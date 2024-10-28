import gzip
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import io
import itertools
import json

def load_json_config(file_path: str) -> dict:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    file_path : str
        The path to the JSON configuration file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def flatten_list(nested_list: list) -> list:
    """
    Flatten a nested list into a single list of elements.

    Parameters
    ----------
    nested_list : list
        A list that may contain nested lists or single elements.

    Returns
    -------
    list
        A flat list containing all elements from the nested list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def dataframe_to_csv(df: pd.DataFrame, compress_data: bool = False) -> bytes:
    """
    Convert a DataFrame to CSV format, optionally compressing it with gzip.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert to CSV.
    compress_data : bool, optional
        If True, compress the CSV data using gzip. Default is False.

    Returns
    -------
    bytes
        The CSV data as bytes. If `compress_data` is True, the data is compressed.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    if compress_data:
        gzipped_csv = gzip.compress(csv_data)
        return gzipped_csv
    else:
        return csv_data

def render_dataframe_as_html(df: pd.DataFrame) -> None:
    """
    Render a DataFrame as HTML in Streamlit, ensuring line breaks are displayed correctly.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to render as HTML.

    Returns
    -------
    None
    """
    # Replace newline characters with <br> tags in string entries
    df_html = df.applymap(
        lambda x: str(x).replace('\n', '<br>') if isinstance(x, str) else x
    ).to_html(escape=False)
    st.markdown(df_html, unsafe_allow_html=True)
