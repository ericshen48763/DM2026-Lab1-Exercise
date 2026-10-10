from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index1: int,
        doc_index2: int
    ) -> Dict[str, Any]:
        """Computes the cosine similarity between two documents using raw counts from the DTM.

        Extracts the feature vectors for the specified document indices from session.feature_matrix
        and computes their cosine similarity. Handles zero-vector edge cases (e.g., empty text documents)
        by returning 0.0 similarity instead of NaN or raising an error.

        Args:
            doc_index1: Integer row index of the first document.
            doc_index2: Integer row index of the second document.

        Returns:
            Dict containing result_id, doc_index1, doc_index2, and similarity.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_index1 < 0 or doc_index1 >= n_docs:
            return {"error": f"doc_index1 ({doc_index1}) is out of bounds for matrix with {n_docs} documents."}
        if doc_index2 < 0 or doc_index2 >= n_docs:
            return {"error": f"doc_index2 ({doc_index2}) is out of bounds for matrix with {n_docs} documents."}

        # Extract rows
        v1 = session.feature_matrix[doc_index1]
        v2 = session.feature_matrix[doc_index2]

        if issparse(v1):
            v1 = v1.toarray().flatten()
        else:
            v1 = np.asarray(v1).flatten()

        if issparse(v2):
            v2 = v2.toarray().flatten()
        else:
            v2 = np.asarray(v2).flatten()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "similarity": round(similarity, 4)
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index1": doc_index1, "doc_index2": doc_index2},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_k: int = 20) -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation matrix for top-variance terms.

        Selects top_k terms from session.feature_matrix by variance, computes their pairwise Pearson
        correlation matrix, builds a annotated heatmap visualization on session.pending_figure, and
        returns the correlation matrix and selected terms.

        Args:
            top_k: Number of highest-variance terms to select (default 20).

        Returns:
            Dict containing result_id, terms list, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        # Convert matrix to dense array for variance and correlation calculations
        dense_matrix = session.feature_matrix.toarray() if issparse(session.feature_matrix) else np.asarray(session.feature_matrix)
        
        # Calculate term variances across all documents
        variances = np.var(dense_matrix, axis=0)
        
        # Select top k terms by variance, descending
        k = min(top_k, len(session.feature_names))
        top_indices = np.argsort(variances)[::-1][:k]
        
        selected_terms = [session.feature_names[i] for i in top_indices]
        selected_matrix = dense_matrix[:, top_indices]
        
        # Compute Pearson correlation matrix across selected features
        # np.corrcoef expects variables as rows, so transpose selected_matrix (docs x features -> features x docs)
        corr_matrix = np.corrcoef(selected_matrix.T)
        
        # Handle single feature edge case if k == 1
        if k == 1:
            corr_matrix = np.array([[1.0]])

        # Replace any potential NaN correlation values (e.g., zero variance) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Create heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.6)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        # Round matrix values to 4 decimal places for clean return structure
        rounded_corr = np.round(corr_matrix, 4).tolist()

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": rounded_corr
        }

        # Optional full_report DataFrame for visualization tools
        report_df = pd.DataFrame(rounded_corr, index=selected_terms, columns=selected_terms)

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_k": top_k},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
