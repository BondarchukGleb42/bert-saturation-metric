import numpy as np
from summarizer import Summarizer
from evaluate import load

def semantic_saturation(full):
    shorted = model(full, min_length=1)
    recall = bertscore.compute(predictions=[full], references=[shorted], lang="en", 
                  model_type="distilbert-base-uncased")["recall"][0]
    precision = len(shorted) / len(full)
    score = (2 * recall * precision) / (recall + precision + 1e-10)
    return score
