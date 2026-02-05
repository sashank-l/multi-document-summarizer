# !pip install datasets rouge-score

############################################
# ----------- Evaluation Module ----------- #
############################################

from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import silhouette_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import warnings
import hdbscan
from transformers import (
    LongformerTokenizer,
    EncoderDecoderModel,
    BartTokenizer,
    BartForConditionalGeneration,
)


# Device for model inference (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_multidoc_dataset():
    candidates = [
        ("multi_news", None),
        ("multi_news", "default"),
        ("cnn_dailymail", "3.0.0"),  # stable working config
        ("cnn_dailymail", "2.0.0"),
        ("xsum", None),
    ]

    for name, config in candidates:
        try:
            # print(f"Trying: {name} {f'({config})' if config else ''} ...") # Keep this print for debugging candidates
            if config:
                ds = load_dataset(name, config)
            else:
                ds = load_dataset(name)

            print(
                f"Loaded dataset for multi-document evaluation: {name} {f'({config})' if config else ''}"
            )
            return ds
        except Exception as e:
            print(
                f"Failed to load dataset {name} {f'({config})' if config else ''}: {e}"
            )

    raise RuntimeError("No compatible summarization dataset could be loaded.")


def load_longtext_dataset():
    candidates = [
        ("ccdv/govreport-summarization", None),
        ("booksum", "chapter_level"),
        ("pubmed_qa", "pqa_labeled"),
    ]

    for name, config in candidates:
        try:
            if config:
                ds = load_dataset(name, config)
            else:
                ds = load_dataset(name)
            print(
                f"Loaded dataset for long text evaluation: {name} {f'({config})' if config else ''}"
            )
            return ds
        except Exception as e:
            print(
                f"Failed to load dataset {name} {f'({config})' if config else ''}: {e}"
            )

    raise RuntimeError("No valid long text summarization dataset available.")


def clustering_labels(embeddings):
    """Perform HDBSCAN clustering on embeddings and return labels."""
    warnings.filterwarnings("ignore")
    embeddings = np.array(embeddings)
    if len(embeddings) < 2:
        raise ValueError(
            "Not enough data points for clustering. At least 2 are required."
        )
    min_cluster_size = min(2, len(embeddings))
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(embeddings)
    return cluster.labels_


def bart_summarizer(text):
    model_name_bart = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name_bart)
    model = BartForConditionalGeneration.from_pretrained(model_name_bart).to(device)

    tokenize_inputs = tokenizer.encode(
        text, return_tensors="pt", max_length=1024, truncation=True
    ).to(device)

    ids_summarization = model.generate(
        tokenize_inputs, num_beams=4, max_length=150, early_stopping=True
    )

    summary_decoded = tokenizer.decode(ids_summarization[0], skip_special_tokens=True)
    return summary_decoded


def longformer_summarizer(text):
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = EncoderDecoderModel.from_pretrained(
        "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16"
    ).to(device)

    inputs = tokenizer(
        text, return_tensors="pt", padding="longest", truncation=True
    ).input_ids.to(device)

    ids_summarization = model.generate(inputs)
    summary_decoded = tokenizer.decode(ids_summarization[0], skip_special_tokens=True)
    return summary_decoded


def longformer_summarizer_long_text(
    text, max_chunk_length=4000, overlap=200, max_summary_length=1024
):

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = EncoderDecoderModel.from_pretrained(
        "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16"
    ).to(device)

    tokens = tokenizer.encode(text)

    if len(tokens) <= max_chunk_length:
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        summary_ids = model.generate(inputs, max_length=max_summary_length)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    chunk_summaries = []

    for i in range(0, len(tokens), max_chunk_length - overlap):
        chunk_tokens = tokens[i : i + max_chunk_length]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        inputs = tokenizer(chunk_text, return_tensors="pt").input_ids.to(device)
        summary_ids = model.generate(inputs, max_length=max_summary_length // 2)
        chunk_summaries.append(
            tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        )

    return " ".join(chunk_summaries)


def summarize_text(text):
    """Select an appropriate summarizer depending on input length."""
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    input_length = len(bart_tokenizer.encode(text))
    if input_length < 1024:
        summary = bart_summarizer(text)
    elif input_length < 4096:
        summary = longformer_summarizer(text)
    else:
        summary = longformer_summarizer_long_text(text)
    return summary


def evaluate_clustering_quality():
    print("\n=== Evaluating Clustering Quality (20 Newsgroups) ===")

    dataset = fetch_20newsgroups(subset="train")
    texts = dataset.data[:200]  # Limit for speed
    labels_true = dataset.target[:200]

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(texts, convert_to_tensor=True)

    sim_scores = []
    for i in range(50):
        a, b = np.random.randint(0, 200, 2)
        sim_scores.append(float(util.cos_sim(embeddings[a], embeddings[b])))

    try:
        cluster_labels = clustering_labels(embeddings.cpu().numpy())
        sil = silhouette_score(embeddings.cpu().numpy(), cluster_labels)
    except Exception as e:
        print(f"Clustering failed: {e}")
        sil = -1

    print(f"Average Random Document Similarity: {np.mean(sim_scores):.4f}")
    print(f"Silhouette Score: {sil:.4f}")


def evaluate_summarization_accuracy():
    print("\n=== Evaluating Summarization Accuracy ===")

    dataset = load_multidoc_dataset()

    # auto-detect which split exists
    split = "test" if "test" in dataset else "validation"
    sampled = dataset[split].select(range(5))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores, rougeL_scores = [], []

    for sample in sampled:
        # auto-detect correct field names
        doc = (
            sample.get("document") or sample.get("article") or sample.get("text") or ""
        )
        ref = (
            sample.get("summary")
            or sample.get("highlights")
            or sample.get("target")
            or ""
        )

        if not doc or not ref:
            print(f"Skipping sample due to missing document or reference: {sample}")
            continue

        # Use summarizer defined below
        try:
            pred = summarize_text(doc)
        except Exception as e:
            print(f"Summarization failed for sample: {e}")
            pred = ""

        scores = scorer.score(ref, pred)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    if not rouge1_scores:
        print("No valid samples to evaluate for summarization accuracy.")
        return

    print(f"\n--- Result ---")
    print(f"ROUGE-1 (F1 Avg): {np.mean(rouge1_scores):.4f}")
    print(f"ROUGE-L (F1 Avg): {np.mean(rougeL_scores):.4f}")


def run_all_evaluations():
    print("\n############# STARTING EVALUATION #############")
    evaluate_clustering_quality()
    evaluate_summarization_accuracy()
    print("############# EVALUATION DONE #############\n")


run_all_evaluations()
