from __future__ import annotations

from src.pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()
    print("Index and structured store built.")
    print(f"Chunks: {len(pipeline.all_chunks)}")
    print(f"Cache dir: {pipeline.cache_dir}")


if __name__ == "__main__":
    main()
