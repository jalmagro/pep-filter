import csv
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
from typing import Dict, List, Set


class EnrichmentAnalyzer:
    """
    A class to load gene annotations from a CSV file and perform enrichment analyses
    on either components or classes for a given set of genes.

    This class expects a CSV file with columns:
    - gene_id
    - component (go term label)
    - class (broader laber)

    The `perform_enrichment_analysis` method allows you to specify which feature to analyze
    ('component' or 'class') and a list of genes of interest. It then performs a Fisher's exact
    test to determine whether certain categories/classes are overrepresented among those genes
    compared to the background set of all genes.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing gene annotations.

    Attributes
    ----------
    gene_to_features : Dict[str, Dict[str, Set[str]]]
        Maps each gene_id to a dictionary with keys "component" and "class",
        each containing a set of features for that gene.
    all_genes : Set[str]
        A set of all gene IDs from the input file.

    Methods
    -------
    perform_enrichment_analysis(feature: str, genes_of_interest: List[str]) -> pd.DataFrame
        Perform Fisher's exact test to identify enriched terms (categories or classes) among
        the input genes_of_interest. Returns a DataFrame with columns:
        component/class, p_value, count_in_list, count_in_bg, total_in_list, total_in_bg,
        fraction_in_list, fraction_in_bg.
    """

    def __init__(self, filename: str):
        self.gene_to_features = {}
        self.all_genes = set()
        self._load_data(filename)

    def _load_data(self, filename: str):
        """
        Load gene annotations from a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file with columns: gene_id, component, class.

        Notes
        -----
        This method initializes `self.gene_to_features` mapping each gene to its categories and classes.
        """
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"gene_id", "component", "class"}
            if not required_cols.issubset(reader.fieldnames):
                raise ValueError(
                    f"The input file must contain the columns: {required_cols}"
                )

            for row in reader:
                gene_id = row["gene_id"]
                self.all_genes.add(gene_id)

                # Initialize if not present
                if gene_id not in self.gene_to_features:
                    self.gene_to_features[gene_id] = {"component": set(), "class": set()}

                cat = row["component"].strip() if row["component"] else ""
                cls = row["class"].strip() if row["class"] else ""
                if cat:
                    for c in cat.split(','):
                        self.gene_to_features[gene_id]["component"].add(c)
                if cls:
                    for c in cls.split(','):
                        self.gene_to_features[gene_id]["class"].add(c)

    def perform_enrichment_analysis(
        self, feature: str, genes_of_interest: List[str]
    ) -> pd.DataFrame:
        """
        Perform enrichment analysis for a given feature (component or class).

        Parameters
        ----------
        feature : str
            Either 'component' or 'class', specifying which feature to analyze.
        genes_of_interest : List[str]
            A list of gene IDs for which you want to test enrichment.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: 
            - component/class: str, the feature value being tested
            - p_value: float, p-value from Fisher's exact test
            - count_in_list: int, how many genes in genes_of_interest have this feature
            - count_in_bg: int, how many genes in the entire background have this feature
            - total_in_list: int, total number of genes in genes_of_interest
            - total_in_bg: int, total number of genes in background
            - fraction_in_list: float, count_in_list / total_in_list
            - fraction_in_bg: float, count_in_bg / total_in_bg

        Raises
        ------
        ValueError
            If `feature` is not 'component' or 'class'.
        """
        if feature not in {"component", "class"}:
            raise ValueError("Invalid feature. Must be one of: 'component', 'class'")

        # Map each feature value to the set of genes that have it
        feature_to_genes = defaultdict(set)
        for gene in self.all_genes:
            for val in self.gene_to_features[gene][feature]:
                feature_to_genes[val].add(gene)

        genes_of_interest_set = set(genes_of_interest)
        total_in_bg = len(self.all_genes)
        total_in_list = len(genes_of_interest_set)

        records = []
        for val, annotated_genes in feature_to_genes.items():
            A = len(
                annotated_genes.intersection(genes_of_interest_set)
            )  # genes in list & have val
            C = len(annotated_genes)  # in bg have val
            B = total_in_list - A
            D = total_in_bg - C

            contingency = [[A, B], [C, D]]
            odds_ratio, p_value = fisher_exact(contingency, alternative="greater")

            fraction_in_list = A / total_in_list if total_in_list > 0 else 0.0
            fraction_in_bg = C / total_in_bg if total_in_bg > 0 else 0.0

            records.append(
                {
                    feature: val,
                    "selected": A,
                    "baseline": C,
                    "fselected": fraction_in_list,
                    "fbaseline": fraction_in_bg,
                    "pvalue": p_value,                
                }
            )

        df = pd.DataFrame(
            records,
            columns=[
                feature,
                "selected",
                "baseline",
                "fselected",
                "fbaseline",
                "pvalue",
            ],
        )
        return df