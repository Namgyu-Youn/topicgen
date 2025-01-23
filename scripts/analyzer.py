import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .topic_list import TOPIC_LIST

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.debug import debug_async_trace, debug_trace
from utils.logger import get_logger

logger = get_logger(__name__)


class TopicAnalyzer:
    def __init__(self):
        try:
            # Initialize basic attributes
            self.device = "cpu"
            self.model_name = "roberta-base"
            self.max_length = 1024

            logger.info(f"Initializing TopicAnalyzer with model: {self.model_name}")

            # Set topic hierarchy before model initialization
            self.topic_hierarchy = TOPIC_LIST
            logger.info("Topic hierarchy loaded")

            # Initialize tokenizer and model
            logger.info("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self._get_num_labels()
            )
            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception:
            logger.exception("Error in TopicAnalyzer initialization")
            raise

    @debug_trace
    def set_device(self, device: str) -> None:
        """Set the device for model inference."""
        self.device = device
        self.model.to(device)

    @debug_trace
    def _get_num_labels(self) -> int:
        """Calculate number of topics for classification."""
        flattened_topics = []
        for main_cat in self.topic_hierarchy.values():
            for sub_cat in main_cat.values():
                flattened_topics.extend(sub_cat)
        return len(flattened_topics)

    @debug_async_trace
    async def generate_topics(self, text: str, category: str, subcategory: str) -> list[dict[str, float]]:
        """Generate topics from text."""
        try:
            logger.info(f"Generating topics for category: {category}, subcategory: {subcategory}")
            all_topics = [topic for subcat in self.topic_hierarchy[category].values() for topic in subcat]

            # Prepare input and move to device
            inputs = self.tokenizer(
                text[:self.max_length],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits[0]).cpu().numpy()

            # Process results
            topics = [
                {"topic": topic, "score": float(score)}
                for topic, score in zip(all_topics, probabilities, strict=False)
                if score > 0.1
            ]

            topics = sorted(topics, key=lambda x: x["score"], reverse=True)[:10]
            logger.info(f"Generated {len(topics)} topics")
            return topics

        except Exception:
            logger.exception("Error generating topics")
            return []