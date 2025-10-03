# Privacy-Aware Decoding (PAD)

## Project Structure

```
PAD/
├── 📁 data/                 # Attack prompts
├── 📁 result/               # Output results
├── 📁 processed/            # Processed data files
├── 📁 corpus/               # Corpus files for retrieval
├── 📁 RetrievalBase/        
├── 🐍 generate.py           # Main generation script
├── 🐍 llm.py                # LLM engine with PAD
├── 🐍 retriever.py          # Retrieval system
├── 🐍 evaluate.py           # Evaluation script
├── 🐍 utils.py              
├── 📄 environment.yml       
└── 📄 .gitignore           
```

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: 3.9 or higher
- **Conda**: For environment management

## 🚀 Usage

### Running Extraction Attacks (Baseline)

```bash
python generate.py \
    --dataset healthcaremagic \
    --model_name EleutherAI/pythia-6.9b \
    --retriever_model BAAI/bge-large-en-v1.5 \
    --temperature 0.2 \
    --max_tokens 256 \
    --output_file result/healthcaremagic/pythia/baseline.json
```

### Running PAD (Privacy-Aware Decoding)

```bash
python generate.py \
    --dataset healthcaremagic \
    --model_name EleutherAI/pythia-6.9b \
    --retriever_model BAAI/bge-large-en-v1.5 \
    --temperature 0.2 \
    --add_noise \
    --epsilon 0.2 \
    --noise_amplification 3.0 \
    --min_sensitivity 0.4 \
    --max_tokens 256 \
    --output_file result/healthcaremagic/pythia/pad.json
```

### 📊 Evaluation

**Evaluate baseline extraction attack**:
```bash
python evaluate.py \
    --input_file result/healthcaremagic/pythia/baseline.json \
    > result/healthcaremagic/pythia/baseline.txt
```

**Evaluate PAD results**:
```bash
python evaluate.py \
    --input_file result/healthcaremagic/pythia/pad.json \
    > result/healthcaremagic/pythia/pad.txt
```