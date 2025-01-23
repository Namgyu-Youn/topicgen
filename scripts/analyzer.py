import os
import sys
from pathlib import Path

import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .topic_list import TOPIC_LIST
from .optimizer import ModelOptimizer

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
            self.onnx_path = Path("models") / "model.onnx"
            self.max_length = 1024

            logger.info(f"Initializing TopicAnalyzer with model: {self.model_name}")

            # Create models directory if it doesn't exist
            self.onnx_path.parent.mkdir(parents=True, exist_ok=True)

            # Set topic hierarchy before model initialization
            self.topic_hierarchy = TOPIC_LIST
            logger.info("Topic hierarchy loaded")

            # Initialize tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            logger.info("Tokenizer loaded successfully")

            # Initialize ONNX session
            logger.info("Loading ONNX model...")
            self.session = self._load_onnx_model()
            logger.info("ONNX model loaded successfully")

            # Model Optimization
            self.optimizer = ModelOptimizer(self.model_path)
            self.optimized_models = self.optimizer.optimize_all()
        except Exception:
            logger.exception("Error in TopicAnalyzer initialization")
            raise

    @debug_trace
    def set_device(self, device: str) -> None:
        """Set the device for model inference."""
        self.device = device
        # Reinitialize ONNX session with new device
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)

    @debug_trace
    def _get_num_labels(self) -> int:
        """Calculate number of topics for classification."""
        flattened_topics = []
        for main_cat in self.topic_hierarchy.values():
            for sub_cat in main_cat.values():
                flattened_topics.extend(sub_cat)
        return len(flattened_topics)

    @debug_trace
    def _convert_to_onnx(self) -> None:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting model to ONNX format...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self._get_num_labels())
        model.eval()

        # Create sample input for ONNX export
        dummy_input = self.tokenizer("Sample text", return_tensors="pt")

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            self.onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info("Model successfully converted to ONNX format")

    @debug_trace
    def _load_onnx_model(self) -> ort.InferenceSession:
        """Load or create ONNX model."""
        try:
            # Convert to ONNX if model doesn't exist
            if not self.onnx_path.exists():
                logger.info("ONNX model not found, converting from PyTorch...")
                self._convert_to_onnx()

            # Initialize ONNX runtime session with appropriate provider
            providers = ['CPUExecutionProvider'] if self.device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ort.InferenceSession(str(self.onnx_path), providers=providers)

        except Exception:
            logger.exception("Error loading ONNX model")
            raise

    @debug_async_trace
    async def generate_topics(self, text: str, category: str, subcategory: str) -> list[dict[str, float]]:
        """Generate topics from text."""
        try:
            logger.info(f"Generating topics for category: {category}, subcategory: {subcategory}")
            all_topics = [topic for subcat in self.topic_hierarchy[category].values() for topic in subcat]

            # Prepare input for inference
            inputs = self.tokenizer(text[: self.max_length], return_tensors="np", padding=True, truncation=True)

            # Run inference with ONNX
            onnx_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
            outputs = self.session.run(None, onnx_inputs)[0]

            # Process results
            probabilities = outputs[0]
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