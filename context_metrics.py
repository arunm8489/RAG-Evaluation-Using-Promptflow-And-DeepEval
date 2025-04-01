from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,ContextualPrecisionMetric,ContextualRecallMetric,ContextualRelevancyMetric
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
def retrieval_metrics(question: str, actual_answer: str, predicted_answer:str, retrieved_chunks: List)->dict:
    """
    pormptflow tool function to call retrieval metrics
    """

    aoai = get_model()
    eval_model = EvalAzureOpenAI(aoai)
    context_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True,
    verbose_mode= False,
    async_mode = False
    )

    context_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True,
    verbose_mode= False,
    async_mode = False
    )
    
    context_relevency_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True,
    verbose_mode= False,
    async_mode = False
    )

    ## context precision expectes expected_output 
    test_case = LLMTestCase(
        input=question,
        actual_output="",
        expected_output=actual_answer,
        retrieval_context=retrieved_chunks
    )

    context_precision_metric.measure(test_case)    
    context_recall_metric.measure(test_case)
    
    test_case_context_relevency = LLMTestCase(
        input=question,
        actual_output="",
        expected_output="",
        retrieval_context=retrieved_chunks
    )
    context_relevency_metric.measure(test_case_context_relevency)

    resp = {}
    resp['contextual_precision_score'] = float(context_precision_metric.score)
    resp['contextual_recall_score'] = float(context_recall_metric.score)
    resp['context_relevency_score'] = float(context_relevency_metric.score)
    return resp





        