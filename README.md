# f4-its
**F4-ITS**: **Fine-grained Feature Fusion for Food Image-Text Search** is a training-free, **vision-language model (VLM)-guided framework** that
significantly improves retrieval performance through enhanced multi-modal feature representations.
Our approach introduces two key contributions: (1) **a uni-directional(and bi-directional) multi-
modal fusion** strategy that combines image embeddings with VLM-generated textual descriptions to
improve query expressiveness, and (2) a novel **feature-based re-ranking mechanism** for top-k retrieval,
leveraging predicted food ingredients to refine results and boost precision.


## Task 1: Single Image-Text Retrieval - Fusion Architecture
<img src="images/f4-its-fusion-arch.png" alt="F4-ITS Fusion Architecture" width=800/>

## Task 2: topk Retrieval - Fusion + Reranking
<img src="images/f4-its-reranker-arch.png" alt="F4-ITS Reranker" width=800/>

## Evaluation Metrics
<img src="images/metrics.png" alt="Evaluation Metrics" width=800/>

## F4-ITS Performance under different fusion settings
<img src="images/fusion-weights.png" alt="Fusion Metrics" width=800/>
