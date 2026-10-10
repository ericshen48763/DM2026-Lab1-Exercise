from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        lowercase: bool = True,
        stop_words: Optional[str] = None,
        min_df: float = 1,
        max_df: float = 1.0,
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from text in session.dataframe.

        Converts raw document text into a matrix of token counts using CountVectorizer,
        populates session.feature_matrix and session.feature_names, stores the vectorizer
        artifact in session.artifacts["count_vectorizer"], updates session labels, and calculates
        matrix sparsity.

        Args:
            lowercase: Whether to convert all characters to lowercase before tokenizing.
            stop_words: Language stop words to remove ('english' or None).
            min_df: Ignore terms that have a document frequency strictly lower than threshold.
            max_df: Ignore terms that have a document frequency strictly higher than threshold.
            ngram_range: Lower and upper boundary of range of n-values for different n-grams to be extracted (e.g., [1, 1]).

        Returns:
            Dict containing result_id, vocabulary_size, n_documents, non_zero, total_elements,
            and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No valid dataframe or 'text' column found in session. Please load a dataset first."}

        # Handle min_df integer vs float conversion if applicable
        parsed_min_df = int(min_df) if min_df >= 1 else min_df
        parsed_max_df = int(max_df) if max_df > 1 else max_df
        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Fill missing text values with empty string to avoid vectorizer errors
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            min_df=parsed_min_df,
            max_df=parsed_max_df,
            ngram_range=ngram_tuple
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "vocabulary_size": n_cols,
            "n_documents": n_rows,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"lowercase": lowercase, "stop_words": stop_words}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 10) -> Dict[str, Any]:
        """Calculates aggregated term frequencies across all documents in the document-term matrix.

        Sums occurrences of each vocabulary word across all documents in session.feature_matrix.

        Args:
            top_n: Number of top most frequent terms to include in the returned summary.

        Returns:
            Dict containing result_id, total_terms, and top_terms frequency dict.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        # Sum term counts across all document rows
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_freqs = {term: int(count) for term, count in zip(session.feature_names, counts)}

        # Sort terms by frequency descending
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))

        session.artifacts["term_frequencies"] = sorted_freqs

        # Get top N terms for summary
        top_terms = dict(list(sorted_freqs.items())[:top_n])

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": sorted_freqs,
            "top_terms": top_terms
        }

        report_df = pd.DataFrame(
            list(sorted_freqs.items()),
            columns=["term", "frequency"]
        )

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap visualization for a positional slice of the document-term matrix.

        Slices the global document-term matrix up to n_documents and n_terms, plots a heatmap
        annotated with term frequencies, sets session.pending_figure, and returns the sliced array summary.

        Args:
            n_terms: Maximum number of terms (columns) to slice from the vocabulary order.
            n_documents: Maximum number of documents (rows) to slice from the dataset row order.

        Returns:
            Dict containing result_id, matrix slice, terms included, n_documents, and n_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        # Slice matrix positional rows and columns
        sliced_matrix = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        sliced_terms = session.feature_names[0:n_terms]
        sliced_docs_count, sliced_terms_count = sliced_matrix.shape

        # Build heatmap visualization
        fig, ax = plt.subplots(figsize=(max(6, sliced_terms_count * 0.8), max(4, sliced_docs_count * 0.5)))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=list(range(sliced_docs_count)),
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (Top {sliced_docs_count} Documents x Top {sliced_terms_count} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "matrix": sliced_matrix.tolist(),
            "terms": sliced_terms,
            "n_documents": sliced_docs_count,
            "n_terms": sliced_terms_count
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
