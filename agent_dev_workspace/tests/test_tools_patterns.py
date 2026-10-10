import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_known_answer_17():
    """
    Tests mine_patterns_tool against known answer #17 in TEST_FIXTURE.md:
    Using algorithm='fpgrowth' and filtering_method='tfidf'.
    - catA keeps 'alpha', min_sup=1..3 finds pattern {'alpha'} with support 3.
    - catB keeps 'gamma', min_sup=1..4 finds pattern {'gamma'} with support 4.
    - filtering_method='variance' or 'term_frequency' keeps 0 terms and returns clean note.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. TFIDF + FPGrowth on catA (min_sup=1)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catA.get("n_patterns_found") == 1
    patterns_catA = res_catA.get("patterns", [])
    assert len(patterns_catA) == 1
    assert set(patterns_catA[0]["itemset"]) == {"alpha"}
    assert patterns_catA[0]["support"] == 3

    # 2. TFIDF + FPGrowth on catB (min_sup=1)
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catB.get("n_patterns_found") == 1
    patterns_catB = res_catB.get("patterns", [])
    assert len(patterns_catB) == 1
    assert set(patterns_catB[0]["itemset"]) == {"gamma"}
    assert patterns_catB[0]["support"] == 4

    # 3. Variance filtering on catA -> zero terms survive
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var.get("patterns") == []
    assert res_var.get("note") == "no terms survived filtering"


def test_mine_patterns_known_answer_20():
    """
    Tests mine_patterns_tool against known answer #20 in TEST_FIXTURE.md:
    Using algorithm='topk' and algorithm='maxfpgrowth'.
    - topk (k=1) returns exactly 1 pattern for catA ({'alpha'}, sup 3) and catB ({'gamma'}, sup 4).
    - maxfpgrowth returns 0 patterns for both categories even at min_sup=1.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. TopK algorithm (k=1)
    res_topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk_catA.get("patterns", [])) == 1
    assert set(res_topk_catA["patterns"][0]["itemset"]) == {"alpha"}
    assert res_topk_catA["patterns"][0]["support"] == 3

    # 2. MaxFPGrowth algorithm (min_sup=1) -> returns 0 patterns
    res_max_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA.get("patterns") == []

    res_max_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catB.get("patterns") == []
