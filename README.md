# QA Documents

Project to answer questions about documents using a fine-tuned model (Mistral 7B).

## Setup

```bash
make setup
```

## Run

```bash
source .venv/bin/activate
python app.py
```


## Assets

### Chosen Model
- Model Fine-tuned using RAFT (Mistral 7B) : https://huggingface.co/isaiasgutierrezcruz/qa_documents_ft_val_raft
- Dataset used: https://huggingface.co/datasets/isaiasgutierrezcruz/qa_documents_val_raft

### Other Models
- Model Fine-tuned using RAG: https://huggingface.co/isaiasgutierrezcruz/qa_documents_ft_val
- Dataset used: https://huggingface.co/datasets/isaiasgutierrezcruz/qa_documents_val
