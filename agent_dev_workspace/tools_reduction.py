import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Optional, Dict, Any, List
from langchain_core.tools import tool

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions to 2D using PCA, t-SNE, or UMAP, and creates a 2D scatter plot.

        Args:
            method: Dimensionality reduction technique to use ('pca', 'tsne', or 'umap').
            perplexity: Perplexity parameter for t-SNE (controls target distance metrics).
            n_neighbors: Number of neighbors parameter for UMAP.
            random_state: Random state seed for reproducible results.

        Returns:
            Dict summarizing the dimensionality reduction process and stored result metadata.
        """
        if session.feature_matrix is not None:
            X = session.feature_matrix
            if hasattr(X, "toarray"):
                X = X.toarray()
        elif session.dataframe is not None:
            num_cols = session.dataframe.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No numeric features found in session.dataframe for dimensionality reduction.")
            X = session.dataframe[num_cols].values
        else:
            raise ValueError("No data available in session.feature_matrix or session.dataframe.")

        n_samples = X.shape[0]
        summary_extra = {}

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            summary_extra["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            # Perplexity must be less than n_samples
            effective_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=random_state,
                init="random"
            )
            coords = reducer.fit_transform(X)
            summary_extra["kl_divergence"] = float(reducer.kl_divergence_)
        elif method == "umap":
            effective_n_neighbors = min(n_neighbors, max(2, n_samples - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=effective_n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        # Store reduced coordinates in artifacts
        session.artifacts[f"reduced_{method}"] = coords

        # Generate 2D Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"]
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=categories, palette="tab10", ax=ax, s=70)
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            categories = session.dataframe["category"]
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=categories, palette="tab10", ax=ax, s=70)
        else:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], ax=ax, s=70)

        ax.set_title(f"2D Projection via {method.upper()}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_samples": n_samples,
            "n_components": 2,
            **summary_extra
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state
            },
            summary=summary,
            full_report={"coordinates": coords.tolist()}
        )

        return summary

    @tool
    def binarize_labels_tool(
        label_column: str = "category_name",
        prefix: str = "cat"
    ) -> Dict[str, Any]:
        """One-hot encodes categorical labels in session.dataframe into binary matrix indicators.

        Args:
            label_column: Target column name in session.dataframe to binarize (default 'category_name').
            prefix: String prefix for one-hot encoded columns (default 'cat').

        Returns:
            Dict containing result_id, n_documents, classes, and encoded_shape.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if label_column not in session.dataframe.columns:
            # Fallback to 'category' if 'category_name' requested but not present
            if label_column == "category_name" and "category" in session.dataframe.columns:
                label_column = "category"
            else:
                return {"error": f"Column '{label_column}' not found in session.dataframe."}

        # One-hot encode using pandas get_dummies
        encoded_df = pd.get_dummies(session.dataframe[label_column], prefix=prefix, dtype=int)
        classes = [str(col) for col in encoded_df.columns]

        # Store encoded result in artifacts
        session.artifacts["binarized_labels"] = encoded_df
        session.artifacts["binarized_label_classes"] = classes

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "label_column": label_column,
            "classes": classes,
            "encoded_shape": list(encoded_df.shape)
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={"label_column": label_column, "prefix": prefix},
            summary=summary,
            full_report=encoded_df
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
