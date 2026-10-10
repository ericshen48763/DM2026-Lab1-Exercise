import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from langchain_core.tools import tool

# Ensure necessary NLTK resource is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def make_tools(session):
    @tool
    def check_missing_tool(column: str = "text") -> Dict[str, Any]:
        """Checks for missing, null, or empty text values in session.dataframe.

        Args:
            column: Name of the text column to check for missing values (default 'text').

        Returns:
            Dict containing result_id, n_documents, n_missing, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if column not in session.dataframe.columns:
            return {"error": f"Column '{column}' not found in session.dataframe."}

        series = session.dataframe[column]

        # Count NaN/None values as well as empty or whitespace-only strings
        is_missing = series.isna() | (series.astype(str).str.strip() == "")
        missing_indices = session.dataframe[is_missing].index.tolist()
        n_missing = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "column": column,
            "n_missing": n_missing,
            "missing_indices": missing_indices
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"column": column},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(
        column: str = "text",
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        When drop=True, uses keep=False to drop every copy of a duplicated row, matching
        Master's drop_duplicates(keep=False) behavior exactly.

        Args:
            column: Name of the column to check for duplicates (default 'text').
            drop: If True, drops all duplicated rows from session.dataframe using keep=False.

        Returns:
            Dict containing result_id, n_duplicates, duplicate_indices, and n_documents_remaining.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if column not in session.dataframe.columns:
            return {"error": f"Column '{column}' not found in session.dataframe."}

        # Identify all duplicated rows (keep=False marks all copies of duplicates as True)
        duplicated_mask = session.dataframe.duplicated(subset=[column], keep=False)
        duplicate_indices = session.dataframe[duplicated_mask].index.tolist()
        n_duplicates = len(duplicate_indices)

        if drop and n_duplicates > 0:
            # Drop all duplicate copies in place using keep=False
            session.dataframe.drop_duplicates(subset=[column], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        n_documents_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "column": column,
            "drop": drop,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "n_documents_remaining": n_documents_remaining
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"column": column, "drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Subsamples n random rows from session.dataframe using pandas df.sample.

        Args:
            n: Number of random rows to sample (default 5).
            random_state: Optional random seed for reproducible sampling.

        Returns:
            Dict containing result_id, n_samples, sampled_indices, and sample_records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        num_to_sample = min(n, len(session.dataframe))
        sampled_df = session.dataframe.sample(n=num_to_sample, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()
        sample_records = sampled_df.to_dict(orient="records")

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_samples": num_to_sample,
            "sampled_indices": sampled_indices,
            "sample_records": sample_records
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
            full_report=sampled_df
        )

        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspects the first n rows of session.dataframe.

        Note: This tool deliberately has no fixed contract or known answer in specs,
        serving as a general exploratory tool.

        Args:
            n: Number of initial rows to inspect (default 5).

        Returns:
            Dict containing result_id, n_inspected, columns, and head_records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        num_to_inspect = min(n, len(session.dataframe))
        head_df = session.dataframe.head(num_to_inspect)
        head_records = head_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_inspected": num_to_inspect,
            "columns": session.dataframe.columns.tolist(),
            "head_records": head_records
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
            full_report=head_df
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """Calculates text_length summary stats overall and by category, and creates a boxplot.

        Computes character length of document text, calculates .describe() statistics overall and
        broken down by category, builds a seaborn boxplot on session.pending_figure, and returns the stats.

        Args:
            text_column: Text column name in session.dataframe (default 'text').
            category_column: Category column name in session.dataframe (default 'category_name').

        Returns:
            Dict containing result_id, overall_stats, category_stats, and text_lengths list.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        # Fallback for category_column if category_name is requested but category is present
        if category_column not in session.dataframe.columns and "category" in session.dataframe.columns:
            category_column = "category"

        # Calculate character length per document
        text_lengths = session.dataframe[text_column].fillna("").astype(str).apply(len)
        session.dataframe["text_length"] = text_lengths

        # Overall summary stats
        overall_desc = text_lengths.describe().to_dict()
        overall_stats = {k: float(round(v, 4)) for k, v in overall_desc.items()}

        # Category summary stats
        category_stats = {}
        if category_column in session.dataframe.columns:
            grouped = session.dataframe.groupby(category_column)["text_length"]
            for cat_name, group in grouped:
                cat_desc = group.describe().to_dict()
                category_stats[str(cat_name)] = {k: float(round(v, 4)) for k, v in cat_desc.items()}

        # Build Seaborn Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in session.dataframe.columns:
            sns.boxplot(data=session.dataframe, x=category_column, y="text_length", hue=category_column, legend=False, ax=ax, palette="Set2")
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=session.dataframe["text_length"], ax=ax, color="lightblue")
            ax.set_title("Overall Text Length Distribution")

        ax.set_ylabel("Text Character Length")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
            "text_lengths": text_lengths.tolist()
        }

        # Full report DataFrame with text lengths and categories
        report_cols = [text_column, "text_length"]
        if category_column in session.dataframe.columns:
            report_cols.append(category_column)
        report_df = session.dataframe[report_cols]

        session.store_result(
            tool_name="describe_data_tool",
            args={"text_column": text_column, "category_column": category_column},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def tokenize_tool(
        column: str = "text",
        lowercase: bool = True
    ) -> Dict[str, Any]:
        """Tokenizes document text into unigrams using nltk.word_tokenize.

        Populates session.dataframe['unigrams'] and session.artifacts['unigrams'] with token lists.

        Args:
            column: Name of the text column to tokenize (default 'text').
            lowercase: Whether to convert text to lowercase before tokenizing (default True).

        Returns:
            Dict containing result_id, n_documents, total_tokens, and tokenized_docs.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if column not in session.dataframe.columns:
            return {"error": f"Column '{column}' not found in session.dataframe."}

        tokenized_docs = []
        for text in session.dataframe[column]:
            if pd.isna(text) or text is None or str(text).strip() == "":
                tokenized_docs.append([])
            else:
                str_text = str(text)
                if lowercase:
                    str_text = str_text.lower()
                tokens = nltk.word_tokenize(str_text)
                tokenized_docs.append(tokens)

        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_tokens = sum(len(doc) for doc in tokenized_docs)

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(tokenized_docs),
            "total_tokens": total_tokens,
            "tokenized_docs": tokenized_docs
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={"column": column, "lowercase": lowercase},
            summary=summary
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in a specified workspace subdirectory.

        Args:
            subdirectory: Subdirectory path relative to workspace root (default '.').

        Returns:
            Dict containing result_id, subdirectory, directories, and files.
        """
        target_path = Path(subdirectory)
        if not target_path.exists():
            return {"error": f"Path '{subdirectory}' does not exist."}

        if not target_path.is_dir():
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories = []
        files = []

        for entry in os.scandir(target_path):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        inspect_data_tool,
        describe_data_tool,
        tokenize_tool,
        list_files_tool
    ]
