# optimizer.py
import os
import torch
import onnx
import tensorrt as trt
import numpy as np
from pathlib import Path
from onnxruntime.transformers import optimizer
from torch.quantization import quantize_dynamic
from typing import Dict, Optional, Any
from logging import getLogger

logger = getLogger(__name__)

class ModelOptimizer:
   def __init__(
       self,
       model_path: Path,
       save_dir: Path = Path("optimized_models"),
       model_type: str = "roberta",
       batch_size: int = 1,
       sequence_length: int = 512
   ):
       self.model_path = model_path
       self.save_dir = save_dir
       self.model_type = model_type
       self.batch_size = batch_size
       self.sequence_length = sequence_length
       self.save_dir.mkdir(exist_ok=True, parents=True)
       logger.info(f"Initializing ModelOptimizer with model: {model_path}")

   def quantize_torch_model(self) -> Dict[str, Any]:
       model = torch.load(self.model_path)
       model.eval()
       logger.info("Starting model quantization")

       quantized_model = quantize_dynamic(
           model,
           {torch.nn.Linear, torch.nn.Conv2d},
           dtype=torch.qint8
       )

       quantized_path = self.save_dir / "quantized_model.pt"
       torch.save(quantized_model.state_dict(), quantized_path)

       size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
       logger.info(f"Quantized model saved ({size_mb:.2f}MB)")

       return {
           "model": quantized_model,
           "path": quantized_path,
           "size_mb": size_mb
       }

   def optimize_onnx(self) -> Dict[str, Any]:
       logger.info("Starting ONNX optimization")
       optimized_model = optimizer.optimize_model(
           str(self.model_path),
           model_type=self.model_type,
           num_heads=12,
           hidden_size=768,
           optimization_options={
               "enable_attention": True,
               "enable_skip_layer_norm": True,
               "enable_embed_layer_norm": True,
               "enable_bias_skip": True,
               "enable_head_pruning": True
           }
       )

       optimized_path = self.save_dir / "optimized_model.onnx"
       optimized_model.save_model_to_file(str(optimized_path))

       size_mb = os.path.getsize(optimized_path) / (1024 * 1024)
       logger.info(f"Optimized ONNX model saved ({size_mb:.2f}MB)")

       return {
           "model": optimized_model,
           "path": optimized_path,
           "size_mb": size_mb
       }

   def build_tensorrt_engine(self) -> Dict[str, Any]:
       logger.info("Starting TensorRT conversion")
       trt_logger = trt.Logger(trt.Logger.WARNING)
       builder = trt.Builder(trt_logger)
       network = builder.create_network(
           1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
       )
       config = builder.create_builder_config()
       config.max_workspace_size = 4 * 1024 * 1024 * 1024

       if builder.platform_has_fast_fp16:
           config.set_flag(trt.BuilderFlag.FP16)
           logger.info("Enabled FP16 precision")

       parser = trt.OnnxParser(network, trt_logger)
       with open(self.model_path, 'rb') as f:
           if not parser.parse(f.read()):
               error_msg = f'Failed to parse ONNX file: {parser.get_error(0).desc()}'
               logger.error(error_msg)
               raise RuntimeError(error_msg)

       profile = builder.create_optimization_profile()
       profile.set_shape(
           "input_ids",
           min=(1, self.sequence_length),
           opt=(self.batch_size, self.sequence_length),
           max=(self.batch_size * 2, self.sequence_length)
       )
       config.add_optimization_profile(profile)

       engine = builder.build_engine(network, config)
       if engine is None:
           error_msg = "Failed to build TensorRT engine"
           logger.error(error_msg)
           raise RuntimeError(error_msg)

       engine_path = self.save_dir / "model.engine"
       with open(engine_path, "wb") as f:
           f.write(engine.serialize())

       size_mb = os.path.getsize(engine_path) / (1024 * 1024)
       logger.info(f"TensorRT engine saved ({size_mb:.2f}MB)")

       return {
           "engine": engine,
           "path": engine_path,
           "size_mb": size_mb
       }

   def optimize_all(self) -> Dict[str, Dict[str, Any]]:
       logger.info("Starting full optimization pipeline")
       results = {
           "quantized": self.quantize_torch_model(),
           "onnx": self.optimize_onnx(),
           "tensorrt": self.build_tensorrt_engine()
       }

       for name, result in results.items():
           logger.info(f"{name} model size: {result['size_mb']:.2f}MB")

       return results

   @staticmethod
   def benchmark_inference(
       model: Any,
       input_shape: tuple,
       num_iterations: int = 100
   ) -> Dict[str, float]:
       input_data = torch.randint(0, 1000, input_shape)
       times = []

       with torch.no_grad():
           for _ in range(num_iterations):
               start = torch.cuda.Event(enable_timing=True)
               end = torch.cuda.Event(enable_timing=True)

               start.record()
               model(input_data)
               end.record()

               torch.cuda.synchronize()
               times.append(start.elapsed_time(end))

       results = {
           "mean_ms": np.mean(times),
           "std_ms": np.std(times),
           "min_ms": np.min(times),
           "max_ms": np.max(times)
       }

       logger.info(f"Inference benchmark results: {results}")
       return results