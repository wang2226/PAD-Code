import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import random
import torch

from retriever import RetrievalDatabaseBuilder
from llm import LLMEngine, RAGPipeline
from langchain_core.documents import Document
from utils import *
from utils import find_all_file, get_encoding_of_file

logging.basicConfig(level=logging.INFO)


def setup_tokenizer_model(name: str, device: str = "auto"):
    """
    Initialize tokenizer and model for text generation.
    
    Args:
        name: HuggingFace model name or path
        device: Device to load model on ("auto", "cpu", "cuda:0", etc.)
    
    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, device_map=device)
    return tokenizer, model


def main():
    """
    Main function that orchestrates the privacy-preserving RAG pipeline.
    
    The pipeline consists of:
    1. Argument parsing and configuration
    2. Dataset-specific corpus loading and preprocessing
    3. Prompt loading
    4. Retrieval database construction
    5. LLM initialization with privacy mechanisms
    6. RAG pipeline execution with privacy tracking
    7. Results saving and logging
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # === Argument Parsing ===
    parser = argparse.ArgumentParser(description="Privacy-Preserving RAG Pipeline")
    
    # Core privacy parameters
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Enable differential privacy noise injection for privacy-preserving decoding."
    )
    parser.add_argument(
        "--noise_amplification",
        type=float,
        default=3.0,
        help="Noise amplification factor for enhanced DP"
    )
    parser.add_argument(
        "--min_sensitivity",
        type=float,
        default=0.4,
        help="Minimum sensitivity bound for enhanced DP (optimal: 0.4)"
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="Privacy epsilon parameter (optimal: 0.2)")
    parser.add_argument("--alpha", type=float, default=10.0, help="RDP alpha parameter for composition (default: 10.0)")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for DP accounting")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Model and system configuration
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-6.9b", help="Language model to use")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: result/rag_results.json)")
    parser.add_argument("--retriever_model", type=str, default="all-MiniLM-L6-v2", help="Retriever embedding model name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cuda:0', 'cuda:7', 'cpu', 'auto')")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["healthcaremagic", "icliniq", "enron_mail"],
        default="healthcaremagic",
        help="Dataset to use: healthcaremagic, icliniq, or enron_mail."
    )
    
    # Advanced DP features
    parser.add_argument(
        "--disable_screening",
        action="store_true",
        help="Disable screening mechanism (skip noise for safe predictions)."
    )
    parser.add_argument(
        "--disable_calibration",
        action="store_true",
        help="Disable data-dependent noise calibration."
    )
    
    # Noise type configuration
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["adaptive", "static"],
        default="adaptive",
        help="Type of noise injection: 'adaptive' (default) or 'static' (uniform noise baseline)"
    )
    parser.add_argument(
        "--static_noise_scale",
        type=float,
        default=0.1,
        help="Noise scale for static baseline (uniform noise injection)"
    )
    

    
    # Corpus preprocessing options
    parser.add_argument(
        "--force_regenerate_corpus",
        action="store_true",
        help="Force regeneration of preprocessed corpus file (for enron_mail dataset)"
    )
    args = parser.parse_args()

    # === Logging Configuration ===
    logging.info("Starting Privacy-Preserving RAG Pipeline")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Retriever: {args.retriever_model}")
    logging.info(f"Device: {args.device}")
    
    # Log privacy configuration
    if args.add_noise:
        if args.noise_type == "static":
            logging.info(f"Static Baseline DP: ε={args.epsilon} | δ={args.delta} | noise_scale={args.static_noise_scale}")
        else:
            logging.info(f"Adaptive DP: ε={args.epsilon} | δ={args.delta}")
            logging.info(f"Features: screening={not args.disable_screening}, calibration={not args.disable_calibration}")
            logging.info(f"Enhancement: amplification={args.noise_amplification}, min_sensitivity={args.min_sensitivity}")
    else:
        logging.info("No noise-based privacy protection enabled")
        

    
    logging.info(f"Generation parameters: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    # === Directory Setup ===
    corpus_dir = "corpus"
    processed_dir = "processed"
    out_dir = "result"

    # === Dataset-Specific Configuration ===
    if args.dataset == "healthcaremagic":
        corpus_file = os.path.join(processed_dir, "healthcaremagic-corpus.json")
        raw_file = os.path.join(corpus_dir, "healthcaremagic-100k.json")
        preprocess_fn = preprocess_healthcaremagic
        input_col = "input"
        output_col = "output"
    elif args.dataset == "icliniq":
        corpus_file = os.path.join(processed_dir, "icliniq-corpus.json")
        raw_file = os.path.join(corpus_dir, "icliniq.json")
        preprocess_fn = preprocess_iclinq
        input_col = "input"
        output_col = "answer_icliniq"
    elif args.dataset == "enron_mail":
        # Special handling for enron_mail dataset
        # Uses raw email files and extracts only email body content
        corpus_file = os.path.join(processed_dir, "enron_mail-corpus.json")
        raw_file = "corpus/enron_mail"  # Directory containing raw email files
        input_col = "content"  # Email body content
        output_col = "content"  # Same content for input/output
        
        # Preprocess enron_mail corpus if needed
        if not os.path.exists(corpus_file) or args.force_regenerate_corpus:
            if not os.path.exists(corpus_file):
                logging.info(f"Preprocessed corpus file {corpus_file} not found. Creating from raw files...")
            elif args.force_regenerate_corpus:
                logging.info(f"Force regeneration requested. Recreating preprocessed corpus from raw files...")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Process all email files and extract body content
            corpus_data = []
            file_paths = list(find_all_file(raw_file))
            for file_path in tqdm(file_paths, desc="Processing enron_mail files"):
                try:
                    encoding = get_encoding_of_file(file_path)
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    
                    if content:  # Only include non-empty files
                        # Extract only email body content (exclude headers)
                        from utils import extract_email_body
                        email_body = extract_email_body(content)
                        
                        if email_body and len(email_body) > 50:  # Filter for substantial content
                            corpus_data.append({
                                "content": email_body,
                                "file_path": file_path
                            })
                except Exception as e:
                    logging.warning(f"Error processing {file_path}: {e}")
                    continue
            
            # Save preprocessed corpus
            with open(corpus_file, "w", encoding="utf-8") as f:
                json.dump(corpus_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Created preprocessed corpus with {len(corpus_data)} documents at {corpus_file}")
        else:
            logging.info(f"Using existing preprocessed corpus file: {corpus_file}")

    # Validate required files exist
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Missing required file/directory: {raw_file}")

    # === Step 1: Load Test Prompts ===
    prompt_file = os.path.join("data", f"{args.dataset}_prompt.json")
    if os.path.exists(prompt_file):
        logging.info(f"Loading test prompts from {prompt_file}")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        df_with_prompts = pd.DataFrame({"prompt": prompts})
        logging.info(f"Loaded {len(prompts)} test prompts from {prompt_file}")
    else:
        logging.error(f"Prompt file {prompt_file} not found.")
        raise FileNotFoundError(f"Required prompt file {prompt_file} not found.")

    # === Step 2: Build Retrieval Database ===
    if args.dataset == "enron_mail":
        # Load preprocessed enron_mail corpus (email bodies only)
        logging.info(f"Loading enron_mail corpus (email body only) from {corpus_file}")
        
        with open(corpus_file, "r", encoding="utf-8") as f:
            raw_corpus = json.load(f)

        # Create documents from email bodies
        documents = [
            Document(page_content=item[input_col], metadata={"output": item[output_col]})
            for item in raw_corpus
            if input_col in item
        ]
        
        logging.info(f"Loaded {len(documents)} email bodies from enron_mail corpus")
        
        # No further splitting needed for email bodies
        split_docs = documents
        
        # Build retrieval database
        builder = RetrievalDatabaseBuilder(device=args.device)
        persist_path = f"./RetrievalBase/{args.dataset}-corpus/{args.retriever_model}"

        db = builder.construct_from_documents(
            documents=split_docs,
            encoder_model_name=args.retriever_model,
            db_name=f"{args.dataset}-corpus/{args.retriever_model}",
        )
    else:
        # Load and process other datasets (healthcaremagic, icliniq)
        logging.info(f"Loading corpus from {raw_file}")
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_corpus = json.load(f)

        # Create documents from JSON data
        documents = [
            Document(page_content=item[input_col], metadata={"output": item[output_col]})
            for item in raw_corpus
            if input_col in item and output_col in item
        ]

        # Split documents into chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        # Build retrieval database
        builder = RetrievalDatabaseBuilder(device=args.device)
        persist_path = f"./RetrievalBase/{args.dataset}-corpus/{args.retriever_model}"

        db = builder.construct_from_documents(
            documents=split_docs,
            encoder_model_name=args.retriever_model,
            db_name=f"{args.dataset}-corpus/{args.retriever_model}",
        )

    # === Step 3: Initialize Language Model ===
    model_name = args.model_name
    tokenizer, model = setup_tokenizer_model(model_name, args.device)

    # Initialize LLM with privacy mechanisms
    llm = LLMEngine(
        model=model,
        tokenizer=tokenizer,
        add_noise=args.add_noise,
        epsilon=args.epsilon,
        alpha=args.alpha,
        delta=args.delta,
        enable_screening=not args.disable_screening,
        enable_calibration=not args.disable_calibration,
        noise_amplification=args.noise_amplification,
        min_sensitivity=args.min_sensitivity,
        noise_type=args.noise_type,
        static_noise_scale=args.static_noise_scale
    )

    # === Step 4: Initialize RAG Pipeline ===
    rag = RAGPipeline(
        retriever=db, 
        llm=llm, 
        device=args.device
    )

    # === Step 5: Execute RAG Pipeline and Generate Responses ===
    results = []
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = os.path.join(out_dir, "rag_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process each test prompt
    for i, prompt in enumerate(
        tqdm(df_with_prompts["prompt"], desc="Generating RAG responses")
    ):
        try:
            # Generate response using RAG pipeline
            result = rag.run(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                # top_p=args.top_p,  # Commented out - using temperature sampling
                # do_sample=False,
            )
            
            # Track privacy loss if noise injection is enabled
            epsilon_dp = llm.get_total_privacy_loss()
            gamma_dp = llm.get_gamma()
            if epsilon_dp is not None:
                print(f"Total ε for this response: {epsilon_dp:.4f}")
                if gamma_dp is not None:
                    print(f"γ (fraction with noise): {gamma_dp:.3f}")
            result["epsilon_dp"] = epsilon_dp
            result["gamma_dp"] = gamma_dp
        except Exception as e:
            # Handle errors gracefully
            result = {
                "question": prompt,
                "context": "",
                "answer": "",
                "error": str(e),
            }

        results.append(result)

        # Display results for monitoring
        print("=== Question ===")
        print(result.get("question"))
        print("=== Retrieved Context ===")
        print(result.get("context", ""))
        print("=== Answer ===")
        print(result.get("answer", ""))
        if "error" in result:
            print("=== ERROR ===")
            print(result["error"])
        print("\n")

        # Save results incrementally to preserve progress
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as save_error:
            print(f"Warning: Failed to save intermediate results: {save_error}")

if __name__ == "__main__":
    main()