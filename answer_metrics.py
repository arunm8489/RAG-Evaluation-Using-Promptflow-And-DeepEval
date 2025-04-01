from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
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
def answer_metrics(question: str, actual_answer: str, predicted_answer:str, retrieved_chunks: List)->dict:
    """
    pormptflow tool function to call answer metrics
    """

    aoai = get_model()
    eval_model = EvalAzureOpenAI(aoai)
    faithfull_metric = FaithfulnessMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
    )

    relevence_metrics = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model,
    include_reason=True
)

    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=retrieved_chunks
    )

    faithfull_metric.measure(test_case) 
    relevence_metrics.measure(test_case)   
    
    resp = {}
    resp['answer_faithfullness'] = float(faithfull_metric.score)
    resp['answer_relevence'] = float(relevence_metrics.score)

    return resp



if __name__=="__main__":


    # question = "Where is eiffel tower located?"
    # actual_answer = "Eiffel tower is a famous landmark located in Paris"
    # predicted_answer = ""
    # retrieved_chunks= ["Eiffel tower is located in Paris","Every year people visit France espetially to see Eiffel tower","Eiffel tower is one of the 7 wonders of the world"]
    # resp = retrieval_metrics(question, actual_answer, predicted_answer, retrieved_chunks)
    # print("Score:   ",resp)

    question = "What is the capital of France?Name of prominent landmark and famous soccer team there?"
    actual_answer="Paris is the capital of France. Eiffel tower is a famous landmark there.PSG is theor famous soccer team"
    predicted_answer = ""
    retrieved_chunks= ["Paris is the capital of France","Every year people visit France espetially to visit paris","PSG and monocco are famous soccer teams there"]
    resp = answer_metrics(question, actual_answer, predicted_answer, retrieved_chunks)
    print("Score:   ",resp)
        