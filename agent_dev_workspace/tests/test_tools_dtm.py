import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    session = SessionState()
    
    # Load dataset into session first using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Initialize student's build_dtm_tool with the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    result = build_dtm_tool.invoke({})
    
    # Known Answer #4 assertions from TEST_FIXTURE.md
    # Vocabulary: ['alpha', 'always', 'beta', 'delta', 'gamma'] -> 5 terms
    assert result["vocabulary_size"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    
    # Sparsity on 8x5 matrix: 21 non-zero cells out of 40 total, sparsity_pct = 47.5
    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)
    
    # Artifact check
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM first as term_frequency_tool relies on session.feature_matrix
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    
    build_dtm_tool.invoke({})
    
    # Run term_frequency_tool
    res = term_frequency_tool.invoke({})
    
    # Assertions according to Known Answer #14
    # Expected total frequencies across all documents:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    
    assert res["term_frequencies"] == expected


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    
    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM first
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    
    build_dtm_tool.invoke({})
    
    # 1. Full Matrix Heatmap check (defaults n_terms=20, n_documents=20)
    full_res = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    
    assert full_res["n_documents"] == 8
    assert full_res["n_terms"] == 5
    assert full_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert full_res["matrix"] == expected_full_matrix
    assert session.pending_figure is not None
    
    # Reset pending figure
    session.pending_figure = None
    
    # 2. Slice check: n_terms=2, n_documents=3
    slice_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    
    assert slice_res["n_documents"] == 3
    assert slice_res["n_terms"] == 2
    assert slice_res["terms"] == ["alpha", "always"]
    assert slice_res["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
