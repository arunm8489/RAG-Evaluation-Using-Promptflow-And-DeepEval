from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deep_eval_model import EvalAzureOpenAI
from deep_eval_model import get_model
from promptflow.tracing import trace
from promptflow.core import tool

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from typing import List

@trace 
@tool
def hallucination_metrics(question: str, actual_answer: str, predicted_answer:str, retrieved_chunks: List)->dict:
    """
    pormptflow tool function to call retrieval metrics
    """

    aoai = get_model()
    eval_model = EvalAzureOpenAI(aoai)
    
    hallucination_metrics = HallucinationMetric(
    threshold=0.5,
    model=eval_model,
    include_reason=False,
    verbose_mode= False,
    async_mode = False
    )

    test_case = LLMTestCase(
            input=question,
            actual_output=predicted_answer,
            context=retrieved_chunks
        )
    resp = {}
    hallucination_metrics.measure(test_case) 
    resp["hallucination_score"] = float(hallucination_metrics.score)
    return resp