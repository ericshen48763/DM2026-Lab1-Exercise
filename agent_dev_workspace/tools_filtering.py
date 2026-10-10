from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Calculates variance for each term across all documents and identifies low-variance terms.

        Args:
            threshold: Minimum variance threshold. Terms with variance less than or equal
                       to this value are flagged for removal.

        Returns:
            A summary dictionary containing result_id, counts of kept/removed features,
            and the lists of kept and removed feature names.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "Document-term matrix not found. Run build_dtm_tool first."}

        # Convert sparse matrix to array if needed and calculate variance per column (term)
        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        variances = np.var(dense_matrix, axis=0)

        # Build DataFrame required by visualize_result_tool:
        # First column: term name, Second column: variance value, sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter terms by threshold
        kept_df = df_report[df_report["variance"] > threshold]
        removed_df = df_report[df_report["variance"] <= threshold]

        kept_features = kept_df["term"].tolist()
        removed_features = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(session.feature_names),
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Computes Pearson correlation between each term's count and a target class label.

        Args:
            target_class: The category label string to test against (one-vs-rest).
            threshold: Absolute correlation threshold. Terms with |r| < threshold are flagged for removal.

        Returns:
            Summary dictionary with result_id, target_class, counts, and lists of kept/removed features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "Document-term matrix not found. Run build_dtm_tool first."}

        if session.labels is None and (session.dataframe is None or "category_name" not in session.dataframe.columns):
            return {"status": "error", "message": "Category labels not found in session."}

        # Retrieve per-document labels matching DTM rows
        if session.labels is not None:
            labels = np.asarray(session.labels)
        else:
            labels = session.dataframe["category_name"].values

        if target_class not in labels:
            return {"status": "error", "message": f"Target class '{target_class}' not found in labels."}

        # Create binary target indicator (1 for target_class, 0 otherwise)
        y = (labels == target_class).astype(float)

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute Pearson correlation r for each feature column against y
        correlations = []
        std_y = np.std(y)
        for i in range(dense_matrix.shape[1]):
            x = dense_matrix[:, i]
            std_x = np.std(x)
            if std_x == 0 or std_y == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r):
                    r = 0.0
            correlations.append(r)

        # Build DataFrame formatted for visualize_result_tool (column 1: term, column 2: pearson_r), sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter features based on absolute threshold
        kept_df = df_report[df_report["pearson_r"].abs() >= threshold]
        removed_df = df_report[df_report["pearson_r"].abs() < threshold]

        kept_features = kept_df["term"].tolist()
        removed_features = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(session.feature_names),
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Computes Spearman rank correlation between each term's count and a target class label.

        Args:
            target_class: The category label string to test against (one-vs-rest).
            threshold: Absolute correlation threshold. Terms with |r| < threshold are flagged for removal.

        Returns:
            Summary dictionary with result_id, target_class, counts, and lists of kept/removed features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "Document-term matrix not found. Run build_dtm_tool first."}

        if session.labels is None and (session.dataframe is None or "category_name" not in session.dataframe.columns):
            return {"status": "error", "message": "Category labels not found in session."}

        # Retrieve per-document labels matching DTM rows
        if session.labels is not None:
            labels = np.asarray(session.labels)
        else:
            labels = session.dataframe["category_name"].values

        if target_class not in labels:
            return {"status": "error", "message": f"Target class '{target_class}' not found in labels."}

        # Create binary target indicator (1 for target_class, 0 otherwise)
        y = (labels == target_class).astype(float)

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute Spearman rank correlation r for each feature column against y
        correlations = []
        for i in range(dense_matrix.shape[1]):
            x = dense_matrix[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                res_r = 0.0
            else:
                res = spearmanr(x, y)
                res_r = float(res.statistic if hasattr(res, "statistic") else res[0])
                if np.isnan(res_r):
                    res_r = 0.0
            correlations.append(res_r)

        # Build DataFrame formatted for visualize_result_tool (column 1: term, column 2: spearman_r), sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter features based on absolute threshold
        kept_df = df_report[df_report["spearman_r"].abs() >= threshold]
        removed_df = df_report[df_report["spearman_r"].abs() < threshold]

        kept_features = kept_df["term"].tolist()
        removed_features = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(session.feature_names),
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
