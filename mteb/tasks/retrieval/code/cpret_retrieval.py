from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_CPRET_DATASET = {
    "path": "coldchair16/CPRet-data",
    "revision": "2629cb44edaa00f48b17ec57942ebe98714662eb",
}
_CPRET_LANGUAGES = ["eng-Latn", "zho-Hans", "jpn-Jpan"]
_CPRET_CITATION = r"""
@misc{deng2025cpretdatasetbenchmarkmodel,
  author = {Han Deng and Yuan Meng and Shixiang Tang and Wanli Ouyang and Xinzhu Ma},
  title = {CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming},
  year = {2025},
  eprint = {2505.12925},
  archivePrefix = {arXiv},
  primaryClass = {cs.SE},
  url = {https://arxiv.org/abs/2505.12925},
}
"""
_CPRET_COMMON = dict(
    reference="https://arxiv.org/abs/2505.12925",
    dataset=_CPRET_DATASET,
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    main_score="ndcg_at_10",
    domains=["Programming", "Written"],
    license="cc-by-nc-4.0",
    dialect=[],
    bibtex_citation=_CPRET_CITATION,
)


class CPRetT2CRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CPRetT2CRetrieval",
        description=(
            "CPRet Text-to-Code retrieval: retrieve accepted solution code for a "
            "competitive programming problem statement. The test set contains "
            "problems from 2023-2024, separated temporally from training data. "
            "Only the T2C test corpus, queries, and qrels are used."
        ),
        eval_langs={"T2C": _CPRET_LANGUAGES},
        date=("2000-01-01", "2024-12-31"),
        task_subtypes=["Code retrieval"],
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Retrieve accepted code solutions for the given competitive programming problem."
        },
        **_CPRET_COMMON,
    )


class CPRetC2CRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CPRetC2CRetrieval",
        description=(
            "CPRet Code-to-Code retrieval: retrieve other accepted implementations "
            "solving the same competitive programming problem as the query code. "
            "The test set contains problems from 2023-2024, separated temporally "
            "from training data. Only the C2C test corpus, queries, and qrels are used."
        ),
        eval_langs={"C2C": _CPRET_LANGUAGES},
        date=("2000-01-01", "2024-12-31"),
        task_subtypes=["Code retrieval"],
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Retrieve code implementations solving the same problem as the given code."
        },
        **_CPRET_COMMON,
    )


class CPRetP2DupRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CPRetP2DupRetrieval",
        description=(
            "CPRet Problem-to-Duplicate retrieval: retrieve duplicate or related "
            "competitive programming problems, with relevance annotations verified "
            "by experienced competitive programmers. Original graded qrels scores "
            "are preserved. Only the P2Dup test corpus, queries, and qrels are used; "
            "P2Dup-train-pairs is excluded. Data collection ended on 2024-12-31."
        ),
        eval_langs={"P2Dup": _CPRET_LANGUAGES},
        date=("2000-01-01", "2024-12-31"),
        task_subtypes=["Duplicate Detection"],
        annotations_creators="expert-annotated",
        sample_creation="found",
        prompt={
            "query": "Retrieve duplicate or closely related competitive programming problems."
        },
        **_CPRET_COMMON,
    )


class CPRetS2FullRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CPRetS2FullRetrieval",
        description=(
            "CPRet Simplified-to-Full retrieval: retrieve the original competitive "
            "programming problem statement from a simplified description. Queries "
            "are human-written simplified descriptions translated into English "
            "using Qwen-2.5-Max; original statements are multilingual. Only the "
            "S2Full test corpus, queries, and qrels are used; S2Full-train-pairs is excluded. "
            "Data collection ended on 2024-12-31."
        ),
        eval_langs={"S2Full": _CPRET_LANGUAGES},
        date=("2000-01-01", "2024-12-31"),
        task_subtypes=["Article retrieval"],
        annotations_creators="derived",
        sample_creation="machine-translated",
        prompt={
            "query": "Retrieve the original competitive programming problem for the simplified description."
        },
        **_CPRET_COMMON,
    )
