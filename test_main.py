import os
from dotenv import load_dotenv

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def test_case():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris",
        retrieval_context=["Paris is the capital of France."]
    )
    assert_test(test_case, [correctness_metric])

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="What is the capital of Canada?",
        actual_output="Ottawa",
        expected_output="Ottawa",
        retrieval_context=["Ottawa is the capital of Canada."]
    )
    assert_test(test_case, [correctness_metric])
