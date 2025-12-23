"""
Evaluate AION-Search on Galaxy Zoo DECaLS: Spiral and Merger NDCG@10
"""

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import torch
from aionsearch import AIONSearchClipModel

load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_grad_enabled(False)


def dcg(r):
    """Compute Discounted Cumulative Gain."""
    return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))


def ndcg_score(relevances, k):
    """Compute NDCG@k."""
    actual_dcg = dcg(relevances[:k])
    ideal_dcg = dcg(np.sort(relevances)[::-1][:k])
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


# Load dataset
print("Loading dataset...")
data = load_dataset("astronolan/gz-decals-embeddings", split="train")
print(f"Total galaxies: {len(data)}")

# Extract all data as numpy arrays first (faster than filtering dataset)
print("Extracting data...")
all_embeddings = np.array(data['aion_search_embedding'])
spiral_fractions = np.array(data['has-spiral-arms_yes_fraction'])
merger_fractions = np.array(data['merging_merger_fraction'])
vote_counts = np.array(data['smooth-or-featured_total-votes'])

# Apply quality filter (votes >= 3) using numpy indexing
print("Applying quality filter (votes >= 3)...")
mask = vote_counts >= 3
all_embeddings = all_embeddings[mask]
spiral_fractions = spiral_fractions[mask]
merger_fractions = merger_fractions[mask]
print(f"Galaxies after filtering: {len(all_embeddings)}")

# Load CLIP model
print("Loading AION-Search CLIP model...")
clip_model = AIONSearchClipModel.from_pretrained(device=device)

# Generate query embeddings
print("Generating query embeddings...")
client = OpenAI()

queries = {
    'spirals': 'visible spiral arms',
    'mergers': 'merging'
}

query_embeddings = {}
for name, text in queries.items():
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    text_embedding = torch.tensor([response.data[0].embedding], device=device)
    query_embeddings[name] = clip_model.text_projector(text_embedding).cpu().numpy().flatten()

# Compute similarities
spiral_similarities = all_embeddings @ query_embeddings['spirals']
merger_similarities = all_embeddings @ query_embeddings['mergers']

# Rank by similarity and compute NDCG@10
k = 10

spiral_indices = np.argsort(spiral_similarities)[::-1]
spiral_relevances_ranked = spiral_fractions[spiral_indices]
spiral_ndcg = ndcg_score(spiral_relevances_ranked, k)

merger_indices = np.argsort(merger_similarities)[::-1]
merger_relevances_ranked = merger_fractions[merger_indices]
merger_ndcg = ndcg_score(merger_relevances_ranked, k)

print(f"\nResults:")
print(f"  Spiral NDCG@{k}: {spiral_ndcg:.4f}")
print(f"  Merger NDCG@{k}: {merger_ndcg:.4f}")
