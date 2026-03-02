"""Tests for the RAGAS evaluation framework."""

import json
from pathlib import Path

import pytest

from src.evaluation.ragas_eval import (
    CaseResult,
    EvaluationResult,
    RAGASEvaluator,
    TestCase,
)


@pytest.fixture()
def test_cases_file(tmp_path: Path) -> Path:
    """Create a test cases JSON file."""
    cases = [
        {
            "question": "What is machine learning?",
            "ground_truth": "Machine learning is a subset of AI.",
        },
        {
            "question": "What is Python?",
            "ground_truth": "Python is a programming language.",
        },
    ]
    file_path = tmp_path / "test_qa.json"
    with open(file_path, "w") as f:
        json.dump(cases, f)
    return file_path


def test_load_test_cases(test_cases_file: Path) -> None:
    """load_test_cases loads cases from JSON."""
    cases = RAGASEvaluator.load_test_cases(str(test_cases_file))
    assert len(cases) == 2
    assert cases[0].question == "What is machine learning?"


def test_load_test_cases_missing_file() -> None:
    """load_test_cases raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RAGASEvaluator.load_test_cases("/nonexistent/file.json")


def test_evaluate_without_pipeline() -> None:
    """Evaluation without pipeline produces empty answers."""
    evaluator = RAGASEvaluator(pipeline=None)
    cases = [
        TestCase(
            question="What is ML?",
            ground_truth="ML is AI.",
        )
    ]
    result = evaluator.evaluate(cases)
    assert result.num_cases == 1
    assert result.case_results[0].generated_answer == ""


def test_evaluate_empty_cases() -> None:
    """Evaluation with no cases returns empty result."""
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate([])
    assert result.num_cases == 0


def test_answer_relevancy_computation() -> None:
    """Answer relevancy computes keyword overlap."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_answer_relevancy(
        "What is machine learning?",
        "Machine learning uses algorithms to learn from data.",
    )
    assert 0.0 <= score <= 1.0
    assert score > 0


def test_answer_relevancy_empty() -> None:
    """Answer relevancy is 0 for empty answer."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_answer_relevancy("Question?", "")
    assert score == 0.0


def test_faithfulness_computation() -> None:
    """Faithfulness measures context coverage."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_faithfulness(
        "ML is powerful", ["ML models are powerful tools"]
    )
    assert 0.0 <= score <= 1.0
    assert score > 0


def test_faithfulness_no_context() -> None:
    """Faithfulness is 0 without context."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_faithfulness("Answer", [])
    assert score == 0.0


def test_context_precision() -> None:
    """Context precision counts relevant contexts."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_context_precision(
        "machine learning",
        ["machine learning is cool", "unrelated text"],
    )
    assert score == 0.5


def test_context_recall() -> None:
    """Context recall measures ground truth coverage."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_context_recall(
        "machine learning AI",
        ["machine learning algorithms"],
    )
    assert 0.0 < score <= 1.0


def test_context_recall_empty() -> None:
    """Context recall is 0 for empty ground truth."""
    evaluator = RAGASEvaluator()
    score = evaluator._compute_context_recall("", [])
    assert score == 0.0


def test_generate_report() -> None:
    """generate_report produces markdown output."""
    evaluator = RAGASEvaluator()
    result = EvaluationResult(
        num_cases=1,
        avg_answer_relevancy=0.8,
        avg_faithfulness=0.7,
        avg_context_precision=0.9,
        avg_context_recall=0.6,
        case_results=[
            CaseResult(
                question="Q?",
                generated_answer="A",
                ground_truth="Expected",
            )
        ],
    )
    report = evaluator.generate_report(result)
    assert "# RAG Evaluation Report" in report
    assert "0.800" in report


def test_save_results(tmp_path: Path) -> None:
    """save_results writes JSON to disk."""
    result = EvaluationResult(num_cases=1)
    output = str(tmp_path / "results.json")
    RAGASEvaluator.save_results(result, output)
    assert Path(output).exists()

    with open(output) as f:
        data = json.load(f)
    assert data["num_cases"] == 1


def test_test_case_dataclass() -> None:
    """TestCase has expected fields."""
    tc = TestCase(
        question="Q",
        ground_truth="A",
        contexts=["ctx"],
    )
    assert tc.question == "Q"
    assert tc.contexts == ["ctx"]


def test_evaluation_result_timestamp() -> None:
    """EvaluationResult sets a timestamp."""
    result = EvaluationResult()
    assert result.timestamp != ""
