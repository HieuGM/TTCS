# Legal RAG MongoDB Hybrid Pipeline

Ung dung CLI Legal RAG cho van ban phap luat Viet Nam. Branch nay da thay pipeline cu bang pipeline moi:

```text
User query
-> optional query generation
-> MongoDB Atlas Vector Search + Atlas Search BM25
-> RRF fusion
-> optional BGE reranker
-> top chunks
-> DeepSeek LLM answer
```

Phan danh gia retrieval chi tinh metric tren chunk/document id, khong can sinh cau tra loi bang LLM.

## Cau Truc Chinh

- `main.py`: CLI de hoi dap.
- `src/generator.py`: format context, goi retrieval pipeline va DeepSeek LLM.
- `src/retrieval_pipeline.py`: hybrid retrieval, RRF, optional BGE rerank.
- `src/pipeline_config.py`: doc cau hinh tu `pipeline_config.env`.
- `pipeline_config.env`: file cau hinh pipeline co the sua truc tiep.
- `evaluate_retrieval_export.py`: export candidate retrieval tu MongoDB cho toan bo ground truth.
- `colab_bge_rerank_eval.py`: rerank candidates bang BGE va tinh retrieval metrics tren Colab/GPU.
- `colab_bge_rerank_eval.ipynb`: notebook Colab toi thieu de chay script rerank/eval.
- `evaluate_reranked_retrieval.py`: tinh metric retrieval offline tu file JSON da rerank san.
- `ground_truth/grounth_truth_record_id.csv`: ground truth chinh, 99 dong.
- `rerank_chunk/reranked_legal_results.json`: vi du file candidates da rerank.

## Requirements

Nen dung Python trong conda env `legal-rag`.

Noi dung day du cua `requirements.txt` hien tai:

```text
langchain
langchain_community
langchain-core
langchain-openai
langchain-mongodb
pymongo
python-dotenv
FlagEmbedding
transformers<5
```

Luu y: `transformers<5` la bat buoc khi dung `FlagEmbedding/FlagReranker`. Neu dung `transformers 5.x`, BGE co the loi:

```text
XLMRobertaTokenizer has no attribute prepare_for_model
```

## Cai Dat

```powershell
conda activate legal-rag
pip install -r requirements.txt
```

Neu muon cai tu dau:

```powershell
conda create -n legal-rag python=3.10 -y
conda activate legal-rag
pip install -r requirements.txt
```

## Bien Moi Truong `.env`

Tao file `.env` o root project. Khong commit file nay.

```env
MONGODB_URI=...
DB_NAME=legal
COLLECTION_NAME=legal_col

OPENAI_API_KEY=...

DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=...
```

`OPENAI_API_KEY` dung cho embedding query. `DEEPSEEK_API_KEY` dung cho query generation neu bat va dung cho cau tra loi cuoi trong pipeline chinh.

## MongoDB Atlas Index

Collection hien dang duoc cau hinh:

```text
Database: legal
Collection: legal_col
Atlas Search index: default
Vector Search index: vector_index
Vector field: embedding
Text field: text
```

Pipeline BM25 chi dung Atlas Search `$search`, khong fallback sang MongoDB `$text`. Neu sai index name hoac field, pipeline se bao loi thay vi am tham chay vector-only.

## Cau Hinh Pipeline

Sua truc tiep file `pipeline_config.env`:

```env
ENABLE_QUERY_GENERATION=false
GENERATED_QUERY_COUNT=3
RRF_TOP_K=100
RERANK_TOP_N=20
MONGODB_TEXT_SEARCH_INDEX=default
MONGODB_TEXT_SEARCH_FIELD=text
BGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ENABLE_LOCAL_BGE_RERANK=true
BGE_USE_FP16=false
RRF_C=60
```

Y nghia cac bien:

- `ENABLE_QUERY_GENERATION`: bat/tat sinh query dong nghia bang DeepSeek.
- `GENERATED_QUERY_COUNT`: so query bo sung. Tong query toi da = query goc + so nay.
- `RRF_TOP_K`: so candidates giu lai sau RRF.
- `RERANK_TOP_N`: so chunk cuoi lay ra sau rerank/de dua vao LLM.
- `MONGODB_TEXT_SEARCH_INDEX`: Atlas Search text index, hien la `default`.
- `MONGODB_TEXT_SEARCH_FIELD`: field text dung cho BM25, hien la `text`.
- `ENABLE_LOCAL_BGE_RERANK`: bat BGE reranker local. CPU-only se cham.
- `BGE_USE_FP16`: nen de `false` khi chay CPU.

Neu chi test nhanh tren CPU, nen giam:

```env
RRF_TOP_K=20
RERANK_TOP_N=5
```

## Chay Pipeline Chinh

Nen chay interactive bang:

```powershell
conda activate legal-rag
python main.py
```

Neu dung `conda run`, them `--no-capture-output` de terminal hien prompt ngay:

```powershell
conda run --no-capture-output -n legal-rag python main.py
```

Khong nen dung:

```powershell
conda run -n legal-rag python main.py
```

vi `conda run` co the buffer output, lam tuong nhu chuong trinh bi treo truoc khi in prompt.

## Danh Gia Retrieval Truoc Rerank

Script nay doc toan bo:

```text
ground_truth/grounth_truth_record_id.csv
```

va export candidates sau MongoDB vector + BM25 + RRF.

Chay full evaluation:

```powershell
conda activate legal-rag
python evaluate_retrieval_export.py
```

Chay debug it dong:

```powershell
python evaluate_retrieval_export.py --limit 2
```

Output nam trong:

```text
logs/retrieval_candidates_<timestamp>/
```

Cac file chinh:

- `candidates.jsonl`: input de dua sang Colab rerank.
- `candidates_detail.json`: chi tiet tung cau hoi.
- `candidate_recall_summary.csv`: metric candidate pool truoc rerank.

Metric local truoc rerank:

- `candidate_hit@100`
- `candidate_recall@100`
- `avg_candidates`
- `evaluated_rows`

## Rerank Va Evaluate Tren Colab

Dung khi muon chay BGE tren GPU.

Upload len Colab:

- `colab_bge_rerank_eval.py`
- `logs/retrieval_candidates_<timestamp>/candidates.jsonl`
- `ground_truth/grounth_truth_record_id.csv`

Chay:

```python
!pip install -U FlagEmbedding "transformers<5"
!python colab_bge_rerank_eval.py \
  --candidates candidates.jsonl \
  --ground-truth grounth_truth_record_id.csv \
  --output-dir rerank_eval_output
```

Output:

- `reranked_results.jsonl`
- `rerank_eval_summary.csv`
- `rerank_eval_detail.json`

Metrics:

- `hit`
- `recall`
- `precision`
- `f1`
- `map`
- `mrr`
- `ndcg`
- `context_precision`
- `context_recall`

K values mac dinh:

```text
1, 3, 5, 10
```

## Danh Gia File Da Rerank San

Neu da co file JSON sau rerank, vi du:

```text
rerank_chunk/reranked_legal_results.json
```

chay offline:

```powershell
conda activate legal-rag
python evaluate_reranked_retrieval.py --input rerank_chunk\reranked_legal_results.json
```

Script nay khong goi MongoDB, OpenAI, DeepSeek, BGE hay LLM. No chi so sanh:

```text
record.groundtruth
vs
record.candidates[*].record_id
```

Output nam trong:

```text
logs/reranked_retrieval_eval_<timestamp>/
```

Gom:

- `rerank_eval_summary.csv`
- `rerank_eval_detail.json`

## Kiem Tra Code

Compile nhanh:

```powershell
conda activate legal-rag
python -m compileall main.py src evaluate_retrieval_export.py evaluate_reranked_retrieval.py colab_bge_rerank_eval.py process_traffic_law.py
```

Kiem tra khong con import pipeline cu:

```powershell
rg "CohereRerank|BM25Retriever|ChatGoogleGenerativeAI|src\\.retriever" src main.py evaluate_retrieval_export.py evaluate_reranked_retrieval.py
```

## Loi Thuong Gap

Neu chay `main.py` ma khong thay prompt:

```powershell
conda run --no-capture-output -n legal-rag python main.py
```

hoac activate env truoc:

```powershell
conda activate legal-rag
python main.py
```

Neu BGE bao loi `prepare_for_model`, cai lai:

```powershell
pip install --no-cache-dir "transformers<5"
```

Neu pipeline bao loi Atlas Search index, kiem tra:

```env
MONGODB_TEXT_SEARCH_INDEX=default
MONGODB_TEXT_SEARCH_FIELD=text
```

va dam bao Atlas Search index `default` da READY/Queryable tren collection `legal_col`.

Neu BGE local qua cham tren CPU, tat reranker hoac giam candidate pool:

```env
ENABLE_LOCAL_BGE_RERANK=false
RRF_TOP_K=20
RERANK_TOP_N=5
```
