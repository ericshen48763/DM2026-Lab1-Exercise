import os
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth_miner
import PAMI.frequentPattern.topk.FAE as topk_miner
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth_miner


def make_tools(session):
    """
    Factory returning tools for pattern mining.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent patterns for a specific category's text using PAMI algorithms.

        Builds a category-local DTM with English stop words removed, filters terms using
        the selected filtering method (variance, tfidf, or term_frequency), and extracts
        frequent patterns using FPGrowth, Top-K, or MaxFPGrowth.

        Args:
            category_name: The category label to filter session.dataframe by.
            filtering_method: Term filtering criterion ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI algorithm to run ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support count (required for "fpgrowth" and "maxfpgrowth").
            k: Top k patterns to keep (required for "topk").

        Returns:
            A dictionary containing result summary and mined patterns.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        res_id = session.next_result_id("patterns")

        # 1. Filter DataFrame by category_name
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": f"No records found for category '{category_name}'."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        texts = df_cat["text"].fillna("").tolist()

        # Build fresh per-category DTM with English stop words removed
        vectorizer = CountVectorizer(stop_words="english")
        dtm_cat = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if dtm_cat.shape[1] == 0:
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dense_dtm = dtm_cat.toarray()

        # 2. Apply chosen filtering method
        if filtering_method in ("variance", "term_frequency"):
            if filtering_method == "variance":
                scores = np.var(dense_dtm, axis=0)
            else:
                scores = np.sum(dense_dtm, axis=0)

            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            # Middle 90% (strictly strictly between 5th and 95th percentile)
            keep_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(dtm_cat)
            scores = np.mean(tfidf_mat.toarray(), axis=0)
            p20 = np.percentile(scores, 20)
            # Top 80% (at or above 20th percentile)
            keep_mask = scores >= p20
        else:
            return {"error": f"Invalid filtering_method: {filtering_method}"}

        kept_terms = feature_names[keep_mask]
        kept_dtm = dense_dtm[:, keep_mask]

        if len(kept_terms) == 0:
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 3. Create transactional database (DF2DB)
        df_trans = pd.DataFrame(kept_dtm, columns=kept_terms)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactions.txt")
            
            # DF2DB converts DataFrame to transactional DB
            conv = db_converter.DF2DB(df_trans)
            conv.convert2TransactionalDatabase(db_file, ">=", 1)

            # 4. Mine patterns using specified algorithm
            patterns_list = []
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'fpgrowth'"}
                miner = fpgrowth_miner.FPGrowth(db_file, minSup=min_sup)
                miner.mine()
                df_patterns = miner.getPatternsAsDataFrame()
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k is required for algorithm 'topk'"}
                miner = topk_miner.FAE(db_file, k=k)
                miner.mine()
                df_patterns = miner.getPatternsAsDataFrame()
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'maxfpgrowth'"}
                miner = maxfpgrowth_miner.MaxFPGrowth(db_file, minSup=min_sup)
                miner.mine()
                df_patterns = miner.getPatternsAsDataFrame()
            else:
                return {"error": f"Invalid algorithm: {algorithm}"}

            if df_patterns is not None and not df_patterns.empty:
                col_pat = [c for c in df_patterns.columns if "pat" in c.lower() or "item" in c.lower()]
                col_sup = [c for c in df_patterns.columns if "sup" in c.lower() or "count" in c.lower()]
                
                pat_col = col_pat[0] if col_pat else df_patterns.columns[0]
                sup_col = col_sup[0] if col_sup else df_patterns.columns[1]

                for _, row in df_patterns.iterrows():
                    pat_val = row[pat_col]
                    sup_val = int(row[sup_col])
                    if isinstance(pat_val, str):
                        itemset = pat_val.strip().split()
                    elif isinstance(pat_val, (list, set, tuple)):
                        itemset = list(pat_val)
                    else:
                        itemset = [str(pat_val)]

                    patterns_list.append({"itemset": itemset, "support": sup_val})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_terms_kept": len(kept_terms),
            "n_patterns_found": len(patterns_list),
            "patterns": patterns_list,
            "artifact_key": artifact_key
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)

        return summary

    return [mine_patterns_tool]
