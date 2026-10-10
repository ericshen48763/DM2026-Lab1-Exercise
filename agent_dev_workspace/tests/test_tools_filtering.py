import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert "result_id" in res
    assert res["total_features"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1

    assert "always" in res["removed_features"]
    for term in ["alpha", "beta", "delta", "gamma"]:
        assert term in res["kept_features"]

    stored_res = session.results_store[res["result_id"]]
    full_report = stored_res["full_report"]

    assert list(full_report.columns[:2]) == ["term", "variance"]
    
    delta_row = full_report[full_report["term"] == "delta"]
    assert not delta_row.empty
    delta_var = delta_row["variance"].values[0]
    assert np.isclose(delta_var, 0.25, atol=1e-3)

    always_row = full_report[full_report["term"] == "always"]
    assert not always_row.empty
    always_var = always_row["variance"].values[0]
    assert np.isclose(always_var, 0.109375, atol=1e-3)


def test_pearson_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert "result_id" in res
    assert res["target_class"] == "catB"
    assert res["total_features"] == 5

    stored_res = session.results_store[res["result_id"]]
    full_report = stored_res["full_report"]

    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Known values from TEST_FIXTURE.md #6 for target_class="catB":
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }

    for term, expected_r in expected_pearson.items():
        row = full_report[full_report["term"] == term]
        assert not row.empty, f"Missing term {term} in full_report"
        actual_r = row["pearson_r"].values[0]
        assert np.isclose(actual_r, expected_r, atol=1e-3), f"For term '{term}', expected r={expected_r}, got {actual_r}"


def test_spearman_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert "result_id" in res
    assert res["target_class"] == "catB"
    assert res["total_features"] == 5

    stored_res = session.results_store[res["result_id"]]
    full_report = stored_res["full_report"]

    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Known values from TEST_FIXTURE.md #6 for target_class="catB" Spearman r:
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    for term, expected_r in expected_spearman.items():
        row = full_report[full_report["term"] == term]
        assert not row.empty, f"Missing term {term} in full_report"
        actual_r = row["spearman_r"].values[0]
        assert np.isclose(actual_r, expected_r, atol=1e-3), f"For term '{term}', expected spearman r={expected_r}, got {actual_r}"
