# -*- coding: utf-8 -*-
from typing import List, Dict
from dataclasses import dataclass
from ..splitters import TokenSplitter

@dataclass
class ClassificationResult:
    intent: str
    confidence: float


@dataclass
class Classification:
    system_prompt = """
    Imagine you are an AI integrated into an advanced add-in for text processing and document research. Your primary role is to assist users by accurately understanding their queries and identifying their primary intent. You must classify each query into one of the specific categories. Importantly, if the certainty of your classification for a query is not high, or if the query does not clearly fit into one of these categories, provide multiple potential classifications along with an honest and accurate estimate of your confidence for each classification. This approach allows acknowledging when a query's classification is uncertain or does not neatly fit into the provided categories.

    Confidence is expressed as a percentage, where a lower percentage indicates higher uncertainty. For queries where you cannot confidently assign a single classification, list the top possible categories with their respective confidence levels. This will offer users a range of possible interpretations and assistance options, enhancing the relevance and usefulness of your support.

    Possible outcomes include:
    - Obtaining information about the cited document
    - Giving instructions
    - Formatting text
    - Translating text
    - Summarizing
    - Expanding on a topic
    - Unsure (use this category when the query's classification is highly uncertain)

    Your ability to provide an accurate assessment of both the classification and the level of confidence, especially in cases of ambiguity, is critical to offering relevant and timely assistance to users dealing with complex document processing and research tasks.
    """

    prompt = """Considering the context and the specified categories of possible outcomes, analyze the following query. If the certainty of your primary classification is not high, provide multiple potential classifications with your honest and accurate estimate of your confidence for each. Reflect accurately on the certainty of your classification, including recognizing when a query might be ambiguous or does not fit neatly into any specified category. 
    
    Query:
    ```
    {query}
    ```
    """
    classifications: List[ClassificationResult]

@dataclass
class Classifier:
    model: str
    assistant_cls: type
    results_cls: type = Classification

    def classify(self, query):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            splitter = TokenSplitter(
                model=assistant.model_name,
                chunk_size=int(assistant.ctx_size * 0.6),
                chunk_overlap=int(assistant.ctx_size * 0.05),
            )
            chunk = next(splitter.chunkify(query))
            classification = assistant.process(query=chunk)
            return classification