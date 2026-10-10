import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")
    
    res = check_missing_tool.invoke({"column": "text"})
    
    # Known answer #2: Exactly 1 row has missing/empty text (row 7, the last row)
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]
    assert res["n_documents"] == 8


def test_check_duplicates_tool_known_answer_3():
    # Test duplicate detection without dropping
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")
    
    # Check duplicates without dropping
    res_check = check_duplicates_tool.invoke({"column": "text", "drop": False})
    
    # Known answer #3: Exactly 1 duplicate pair (rows 5 and 6 are duplicates of each other)
    assert res_check["n_duplicates"] == 2
    assert res_check["duplicate_indices"] == [5, 6]
    assert res_check["n_documents_remaining"] == 8

    # Test dropping duplicates with keep=False in a separate isolated session
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")
    
    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in data_tools_drop if t.name == "check_duplicates_tool")
    
    res_drop = check_duplicates_tool_drop.invoke({"column": "text", "drop": True})
    
    # Dropping with keep=False leaves exactly 6 rows (both rows 5 and 6 are removed)
    assert res_drop["n_duplicates"] == 2
    assert res_drop["n_documents_remaining"] == 6
    assert len(session_drop.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")
    
    # Known answer #13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    
    assert res["n_samples"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")
    
    res = inspect_data_tool.invoke({"n": 3})
    
    assert res["n_inspected"] == 3
    assert len(res["head_records"]) == 3
    assert res["head_records"][0]["text"] == "always alpha alpha alpha beta"


def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")
    
    res = describe_data_tool.invoke({"text_column": "text", "category_column": "category_name"})
    
    # Known Answer #18 text lengths per doc: [29, 23, 17, 30, 24, 18, 18, 0]
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert res["text_lengths"] == expected_lengths
    
    # Overall describe stats checks: count=8, mean=19.875, std=9.4330, min=0, max=30
    overall = res["overall_stats"]
    assert overall["count"] == 8.0
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0.0
    assert overall["max"] == 30.0
    
    # Per-category describe stats checks:
    # catA (rows 0, 1, 2, 7): count=4, mean=17.25, std=12.5, min=0, max=29
    catA = res["category_stats"]["catA"]
    assert catA["count"] == 4.0
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["min"] == 0.0
    assert catA["max"] == 29.0
    
    # catB (rows 3, 4, 5, 6): count=4, mean=22.5, std=5.7446, min=18, max=30
    catB = res["category_stats"]["catB"]
    assert catB["count"] == 4.0
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18.0
    assert catB["max"] == 30.0
    
    # Pending figure check
    assert session.pending_figure is not None


def test_tokenize_tool_known_answer_10():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    
    res = tokenize_tool.invoke({"column": "text", "lowercase": True})
    
    # Known Answer #10: One token list per document
    docs = res["tokenized_docs"]
    assert len(docs) == 8
    
    # Row 0: ['always', 'alpha', 'alpha', 'alpha', 'beta']
    assert docs[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    
    # Row 7: empty list
    assert docs[7] == []
    
    # Verify unigrams artifact and dataframe column
    assert "unigrams" in session.artifacts
    assert session.dataframe["unigrams"].tolist() == docs


def test_list_files_tool_known_answer_11():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")
    
    # Known Answer #11: listing newdataset subdirectory
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
