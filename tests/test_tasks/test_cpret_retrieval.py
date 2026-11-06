from unittest.mock import patch

import pytest
import pytrec_eval
from datasets import Dataset

import mteb

pytestmark = pytest.mark.core

_TASKS = [
    ("CPRetT2CRetrieval", "T2C"),
    ("CPRetC2CRetrieval", "C2C"),
    ("CPRetP2DupRetrieval", "P2Dup"),
    ("CPRetS2FullRetrieval", "S2Full"),
]
_LOADER_MODULE = "mteb.abstasks.retrieval_dataset_loaders"


@pytest.mark.parametrize(("name", "subset"), _TASKS)
def test_cpret_loads_only_test_retrieval_configs(name: str, subset: str):
    task = mteb.get_task(name)
    configs = [
        f"{prefix}-{section}"
        for _, prefix in _TASKS
        for section in ("corpus", "queries", "qrels")
    ] + ["PCPCD", "P2Dup-train-pairs", "S2Full-train-pairs"]
    data = {
        f"{subset}-corpus": Dataset.from_dict(
            {"_id": ["d0", "d1", "d2"], "text": ["first", "second", "third"]}
        ),
        f"{subset}-queries": Dataset.from_dict(
            {"_id": ["q0", "unjudged"], "text": ["query", "no judgments"]}
        ),
        f"{subset}-qrels": Dataset.from_dict(
            {
                "query-id": ["q0", "q0", "q0"],
                "corpus-id": ["d0", "d1", "d2"],
                "score": [1, 2, 3],
            }
        ),
    }
    with (
        patch(f"{_LOADER_MODULE}.get_dataset_config_names", return_value=configs),
        patch(
            f"{_LOADER_MODULE}.get_dataset_split_names", return_value=["train", "test"]
        ),
        patch(
            f"{_LOADER_MODULE}.load_dataset",
            side_effect=lambda path, config, **kwargs: data[config],
        ) as load_dataset,
    ):
        task.load_data()
        task.load_data()

    assert task.data_loaded
    assert task.hf_subsets == [subset]
    assert task.eval_splits == ["test"]
    assert task.metadata.main_score == "ndcg_at_10"
    assert 10 in task.k_values
    assert set(task.dataset) == {subset}
    assert set(task.dataset[subset]) == {"test"}
    assert load_dataset.call_count == 3
    assert {call.args[1] for call in load_dataset.call_args_list} == set(data)
    for call in load_dataset.call_args_list:
        assert call.args[0] == "coldchair16/CPRet-data"
        assert call.kwargs["split"] == "test"
        assert call.kwargs["revision"] == task.metadata.dataset["revision"]
        assert call.kwargs["trust_remote_code"] is False

    split_data = task.dataset[subset]["test"]
    assert list(split_data["corpus"]["id"]) == ["d0", "d1", "d2"]
    assert list(split_data["queries"]["id"]) == ["q0"]
    assert split_data["relevant_docs"] == {"q0": {"d0": 1, "d1": 2, "d2": 3}}
    assert split_data["top_ranked"] is None

    evaluator = pytrec_eval.RelevanceEvaluator(
        split_data["relevant_docs"], {"ndcg_cut.10"}
    )
    ideal = evaluator.evaluate({"q0": {"d0": 1.0, "d1": 2.0, "d2": 3.0}})
    reversed_ranking = evaluator.evaluate({"q0": {"d0": 3.0, "d1": 2.0, "d2": 1.0}})
    assert ideal["q0"]["ndcg_cut_10"] == pytest.approx(1.0)
    assert reversed_ranking["q0"]["ndcg_cut_10"] < 1.0


def test_cpret_benchmark_contains_four_test_tasks():
    benchmark = mteb.get_benchmark("CPRet")
    assert {task.metadata.name for task in benchmark.tasks} == {
        name for name, _ in _TASKS
    }
    assert len(benchmark.tasks) == 4
    for task in benchmark.tasks:
        assert task.eval_splits == ["test"]
        assert task.metadata.main_score == "ndcg_at_10"
        task.metadata._validate_metadata()
        stats = task.metadata.descriptive_stats
        assert stats is not None
        assert set(stats) == {"test"}
        relevant_stats = stats["test"]["relevant_docs_statistics"]
        assert relevant_stats["num_missing_query_ids"] == 0
        assert relevant_stats["num_missing_corpus_ids"] == 0
