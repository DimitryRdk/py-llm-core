# -*- coding: utf-8 -*-
from typing import List, Dict
from dataclasses import dataclass
from ..splitters import TokenSplitter


@dataclass
class Classification:
    system_prompt = """
    Imagine you are an AI integrated into an advanced add-in for text processing and document research. Your primary role is to assist users by accurately understanding their queries and identifying their primary intent. After analyzing the query, you must determine the most appropriate classification based on the content and context of the request. You will classify each query by marking one, and only one, of the specific categories as True, which best represents the primary intent of the query.

    It is crucial to provide an honest and accurate assessment of the classification. If the query's classification is uncertain or does not neatly fit into the provided categories, you should mark the 'unsure' category as True. This approach acknowledges when a query's classification is uncertain or ambiguous, offering users a range of possible interpretations and assistance options, enhancing the relevance and usefulness of your support.

    Possible outcomes (mark one as True):
    - obtaining_simple_information: for queries seeking straightforward information extraction from the documents.
    - generating_text_from_information: for queries that require generating new text based on the information provided.
    - formatting_text: for requests related to text formatting, such as applying styles or adjusting layouts.
    - translating_text: for translation requests between languages.
    - summarizing: for summarization requests to condense information.
    - expanding: for queries asking to expand on a topic with more details.
    - correcting_information: for requests that seek correction or clarification of misinformation or unclear details.
    - generating_content: for requests aimed at generating new content based on specified criteria.
    - unsure: use this category when the query's classification is highly uncertain or does not fit neatly into any specified category.

    Your ability to accurately classify the query, reflecting on the certainty of your classification and recognizing when a query might be ambiguous or does not fit neatly into any specified category, is critical to offering relevant and timely assistance to users dealing with complex document processing and research tasks.
    """

    prompt = """Considering the context and the specified categories of possible outcomes, analyze the following query. Determine the primary intent and mark the corresponding category as True. If the query might be ambiguous or does not fit neatly into any specified category, mark 'unsure' as True.

    Query:
    ```
    {query}
    ```
    """
    obtaining_simple_information: bool = False
    generating_text_from_information: bool = False
    giving_instructions: bool = False
    formatting_text: bool = False
    translating_text: bool = False
    summarizing: bool = False
    expanding: bool = False
    correcting_information: bool = False
    generating_content: bool = False
    unsure: bool = False

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