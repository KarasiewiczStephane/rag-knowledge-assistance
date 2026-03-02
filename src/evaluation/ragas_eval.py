"""RAGAS evaluation framework for measuring RAG pipeline quality."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case.

    Attributes:
        question: The input question.
        ground_truth: Expected correct answer.
        contexts: Optional list of expected context strings.
    """

    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Result of evaluating a single test case.

    Attributes:
        question: The evaluated question.
        generated_answer: The pipeline's answer.
        ground_truth: Expected answer.
        contexts: Retrieved context strings.
        answer_relevancy: How relevant the answer is (0-1).
        faithfulness: How faithful to the context (0-1).
        context_precision: Ratio of relevant retrieved chunks.
        context_recall: How much ground truth was covered.
    """

    question: str
    generated_answer: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


@dataclass
class EvaluationResult:
    """Aggregate evaluation results.

    Attributes:
        num_cases: Number of test cases evaluated.
        avg_answer_relevancy: Mean answer relevancy score.
        avg_faithfulness: Mean faithfulness score.
        avg_context_precision: Mean context precision.
        avg_context_recall: Mean context recall.
        case_results: Per-case detailed results.
        timestamp: When the evaluation was run.
    """

    num_cases: int = 0
    avg_answer_relevancy: float = 0.0
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    case_results: list[CaseResult] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RAGASEvaluator:
    """Evaluates RAG pipeline quality using custom RAGAS-style metrics.

    Computes answer relevancy, faithfulness, context precision,
    and context recall on a set of test Q&A pairs.

    Args:
        pipeline: The RAG pipeline to evaluate. Can be None for
            offline evaluation.
    """

    def __init__(self, pipeline: Any = None) -> None:
        self._pipeline = pipeline

    @staticmethod
    def load_test_cases(file_path: str) -> list[TestCase]:
        """Load test cases from a JSON file.

        Args:
            file_path: Path to the JSON test cases file.

        Returns:
            List of TestCase objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Test cases file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        cases = []
        for item in data:
            cases.append(
                TestCase(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    contexts=item.get("contexts", []),
                )
            )

        logger.info("Loaded %d test cases from %s", len(cases), path)
        return cases

    def evaluate(self, test_cases: list[TestCase]) -> EvaluationResult:
        """Run evaluation on a set of test cases.

        For each test case, queries the pipeline and computes
        metrics by comparing the generated answer against the
        ground truth and retrieved contexts.

        Args:
            test_cases: List of TestCase objects to evaluate.

        Returns:
            EvaluationResult with aggregate and per-case metrics.
        """
        case_results: list[CaseResult] = []

        for tc in test_cases:
            generated_answer = ""
            contexts: list[str] = []

            if self._pipeline is not None:
                try:
                    response = self._pipeline.process_query(tc.question)
                    generated_answer = response.answer
                    contexts = [c.excerpt for c in response.citations.citations]
                except Exception as e:
                    logger.error(
                        "Error evaluating '%s': %s",
                        tc.question,
                        e,
                    )
                    generated_answer = f"Error: {e}"

            relevancy = self._compute_answer_relevancy(tc.question, generated_answer)
            faithfulness = self._compute_faithfulness(generated_answer, contexts)
            precision = self._compute_context_precision(tc.ground_truth, contexts)
            recall = self._compute_context_recall(tc.ground_truth, contexts)

            case_results.append(
                CaseResult(
                    question=tc.question,
                    generated_answer=generated_answer,
                    ground_truth=tc.ground_truth,
                    contexts=contexts,
                    answer_relevancy=relevancy,
                    faithfulness=faithfulness,
                    context_precision=precision,
                    context_recall=recall,
                )
            )

        result = self._aggregate_results(case_results)
        logger.info(
            "Evaluation complete: %d cases, avg_relevancy=%.3f, avg_faithfulness=%.3f",
            result.num_cases,
            result.avg_answer_relevancy,
            result.avg_faithfulness,
        )
        return result

    def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """Compute answer relevancy based on keyword overlap.

        Args:
            question: The original question.
            answer: The generated answer.

        Returns:
            Relevancy score between 0 and 1.
        """
        if not answer or not question:
            return 0.0

        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "do",
            "does",
            "did",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "or",
            "not",
            "it",
        }
        q_keywords = q_words - stopwords
        if not q_keywords:
            return 0.5

        overlap = q_keywords & a_words
        return min(len(overlap) / max(len(q_keywords), 1), 1.0)

    def _compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Compute faithfulness based on context coverage.

        Args:
            answer: The generated answer.
            contexts: Retrieved context strings.

        Returns:
            Faithfulness score between 0 and 1.
        """
        if not answer or not contexts:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words: set[str] = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())

        if not answer_words:
            return 0.0

        overlap = answer_words & context_words
        return min(len(overlap) / max(len(answer_words), 1), 1.0)

    def _compute_context_precision(
        self, ground_truth: str, contexts: list[str]
    ) -> float:
        """Compute context precision: relevant contexts / total.

        Args:
            ground_truth: Expected answer text.
            contexts: Retrieved context strings.

        Returns:
            Precision score between 0 and 1.
        """
        if not contexts:
            return 0.0

        gt_words = set(ground_truth.lower().split())
        relevant = sum(1 for ctx in contexts if gt_words & set(ctx.lower().split()))
        return relevant / len(contexts)

    def _compute_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Compute context recall: covered GT words / total GT words.

        Args:
            ground_truth: Expected answer text.
            contexts: Retrieved context strings.

        Returns:
            Recall score between 0 and 1.
        """
        if not ground_truth:
            return 0.0

        gt_words = set(ground_truth.lower().split())
        if not gt_words:
            return 0.0

        context_words: set[str] = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())

        covered = gt_words & context_words
        return len(covered) / len(gt_words)

    def _aggregate_results(self, case_results: list[CaseResult]) -> EvaluationResult:
        """Aggregate per-case results into overall metrics.

        Args:
            case_results: List of per-case results.

        Returns:
            EvaluationResult with averages.
        """
        n = len(case_results)
        if n == 0:
            return EvaluationResult()

        return EvaluationResult(
            num_cases=n,
            avg_answer_relevancy=sum(c.answer_relevancy for c in case_results) / n,
            avg_faithfulness=sum(c.faithfulness for c in case_results) / n,
            avg_context_precision=sum(c.context_precision for c in case_results) / n,
            avg_context_recall=sum(c.context_recall for c in case_results) / n,
            case_results=case_results,
        )

    def generate_report(self, result: EvaluationResult) -> str:
        """Generate a human-readable evaluation report.

        Args:
            result: EvaluationResult to format.

        Returns:
            Formatted report string.
        """
        lines = [
            "# RAG Evaluation Report",
            f"**Timestamp:** {result.timestamp}",
            f"**Test Cases:** {result.num_cases}",
            "",
            "## Aggregate Metrics",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Answer Relevancy | {result.avg_answer_relevancy:.3f} |",
            f"| Faithfulness | {result.avg_faithfulness:.3f} |",
            f"| Context Precision | {result.avg_context_precision:.3f} |",
            f"| Context Recall | {result.avg_context_recall:.3f} |",
            "",
            "## Per-Case Results",
        ]

        for i, case in enumerate(result.case_results, 1):
            lines.append(f"\n### Case {i}")
            lines.append(f"**Q:** {case.question}")
            lines.append(f"**Expected:** {case.ground_truth}")
            lines.append(f"**Generated:** {case.generated_answer}")
            lines.append(
                f"Relevancy={case.answer_relevancy:.3f} | "
                f"Faithfulness={case.faithfulness:.3f} | "
                f"Precision={case.context_precision:.3f} | "
                f"Recall={case.context_recall:.3f}"
            )

        return "\n".join(lines)

    @staticmethod
    def save_results(result: EvaluationResult, file_path: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            result: EvaluationResult to save.
            file_path: Output file path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Saved evaluation results to %s", path)
