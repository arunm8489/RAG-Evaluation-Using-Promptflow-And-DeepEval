from promptflow.core import tool
from promptflow.tracing import trace
from typing import List,Dict
from promptflow.core import log_metric



@trace
@tool 
def aggregate(retrieval_metrics,answer_metrics,hallucination_metrics
              )-> Dict[str, float]:
    
    """
    score aggregation node
    """
    
    context_precision_scores = [i["contextual_precision_score"] for i in retrieval_metrics]
    context_recall_scores = [i["contextual_recall_score"] for i in retrieval_metrics]
    context_relevency_scores = [i["context_relevency_score"] for i in retrieval_metrics]

    answer_faithfullness_score = [i['answer_faithfullness'] for i in answer_metrics]
    answer_relevency_score = [i['answer_relevence'] for i in answer_metrics]
    
    
    hallucination_score = [i['hallucination_score'] for i in hallucination_metrics]
    
    aggregated_context_precision = sum(context_precision_scores)/len(context_precision_scores)
    aggregated_context_recall = sum(context_recall_scores)/len(context_recall_scores)
    aggregated_context_relevency = sum(context_relevency_scores)/len(context_relevency_scores)
    aggregated_answer_faithfullness = sum(answer_faithfullness_score)/len(answer_faithfullness_score)
    aggregated_answer_relevence = sum(answer_relevency_score)/len(answer_relevency_score)
    aggregated_hallucination_score = sum(hallucination_score)/len(hallucination_score)

    log_metric(key="aggregated_context_precision",value=aggregated_context_precision)
    log_metric(key="aggregated_context_recall",value=aggregated_context_recall),
    log_metric(key="aggregated_context_relevency",value=aggregated_context_relevency),
    log_metric(key="aggregated_answer_faithfullness",value=aggregated_answer_faithfullness)
    log_metric(key="aggregated_answer_relevence",value=aggregated_answer_relevence),
    log_metric(key="aggregated_hallucination_score",value=aggregated_hallucination_score)
            

    return {"aggregated_context_precision":aggregated_context_precision,
            "aggregated_context_recall":aggregated_context_recall,
            "aggregated_context_relevency":aggregated_context_relevency,
            "aggregated_answer_faithfullness":aggregated_answer_faithfullness,
            "aggregated_answer_relevence":aggregated_answer_relevence,
            "aggregated_hallucination_score":aggregated_hallucination_score
            }