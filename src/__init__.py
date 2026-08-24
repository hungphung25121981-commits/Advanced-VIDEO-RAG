"""video-visual-rag: Video-Native Multimodal Hybrid RAG System.

Single-model (Qwen2.5-VL-8B-Instruct, 4-bit NF4) pipeline covering:
  Phase 1 - Keyframe extraction (OpenCV SSIM + PySceneDetect)
  Phase 2 - Visual captioning & OCR (PaddleOCR + Qwen2.5-VL)
  Phase 3 - Hybrid indexing (FAISS dense + BM25 sparse, bge-m3 embeddings)
  Phase 4 - Orchestration (RRF fusion, multi-hop sub-query decomposition,
            timestamp-grounded answer synthesis) via Qwen2.5-VL
"""

__version__ = "0.1.0"
