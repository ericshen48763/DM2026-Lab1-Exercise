import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    
    return session


def test_reduce_dimensions_pca():
    session = setup_fixture_session()
    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    summary = reduce_tool.invoke({"method": "pca"})

    # Known answer 7 expected coordinates
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_explained_variance = [0.7532, 0.1965]

    coords = session.artifacts["reduced_pca"]

    assert coords.shape == (8, 2)
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(summary["explained_variance_ratio"], expected_explained_variance, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    session = setup_fixture_session()
    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    summary = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    coords = session.artifacts["reduced_tsne"]

    assert coords.shape == (8, 2)
    assert "kl_divergence" in summary
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    session = setup_fixture_session()
    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    summary = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    # Known answer 9 expected coordinates
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    coords = session.artifacts["reduced_umap"]

    assert coords.shape == (8, 2)
    assert np.allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    session = setup_fixture_session()
    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    summary = binarize_tool.invoke({"label_column": "category_name", "prefix": "cat"})

    assert summary["n_documents"] == 8
    assert summary["encoded_shape"] == [8, 2]
    assert "cat_catA" in summary["classes"]
    assert "cat_catB" in summary["classes"]

    encoded_df = session.artifacts["binarized_labels"]
    assert encoded_df.shape == (8, 2)

    # First 3 rows and row 7 are catA, rows 3-6 are catB in sample_fixture.csv
    assert encoded_df["cat_catA"].tolist() == [1, 1, 1, 0, 0, 0, 0, 1]
    assert encoded_df["cat_catB"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
