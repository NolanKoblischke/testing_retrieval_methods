"""
AION-Search Demo: Text and Image Search over Galaxy Embeddings

This script demonstrates two search capabilities:
1. Image-to-Image: Find the most similar galaxy to a query image
2. Text-to-Image: Find galaxies matching a natural language description

Uses pre-computed AION-Search embeddings from the astronolan/gz-decals-embeddings dataset.
Requires an OpenAI API key in .env for text embeddings.
"""

import torch
import numpy as np
from datasets import load_dataset
from astropy.io import fits
from openai import OpenAI
from dotenv import load_dotenv
from aion.modalities import LegacySurveyImage
from aion.codecs import CodecManager
from aion.model import AION
from aionsearch import AIONSearchClipModel

load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_grad_enabled(False)

# Load data and models
train_data = load_dataset("astronolan/gz-decals-embeddings", split="train[:5%]")
all_embeddings = np.array(train_data['aion_search_embedding'])
print(f"Loaded {len(train_data)} galaxies (5% of total in astronolan/gz-decals-embeddings dataset)")

aion_model = AION.from_pretrained("polymathic-ai/aion-base").to(device).eval()
codec_manager = CodecManager(device=device)
clip_model = AIONSearchClipModel.from_pretrained(device=device)

# --- Image Search ---
print("\n[Image Search]")
query_url = "https://www.legacysurvey.org/viewer/cutout.fits?ra=69.0111&dec=-35.9962&layer=ls-dr10&pixscale=0.262"
print(f"Query: {query_url.replace('fits', 'jpg')}")

query_image = fits.getdata(query_url)
image_flux = torch.tensor(query_image.astype("float32")).unsqueeze(0).to(device)
image = LegacySurveyImage(flux=image_flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
query_embedding = aion_model.encode(codec_manager.encode(image), num_encoder_tokens=600).mean(axis=1)
query_proj = clip_model.image_projector(query_embedding).cpu().numpy().flatten()

similarities = all_embeddings @ query_proj
best_idx = np.argmax(similarities)
match = train_data[best_idx]

print(f"\nBest match: RA={match['ra']:.4f}, Dec={match['dec']:.4f}")
print(f"  Spiral arms: {match['has-spiral-arms_yes']}/{match['has-spiral-arms_total-votes']} human votes, Cosine similarity: {similarities[best_idx]:.4f}")
print(f"  https://www.legacysurvey.org/viewer/cutout.jpg?ra={match['ra']}&dec={match['dec']}&layer=ls-dr10&pixscale=0.262")

# --- Text Search ---
print("\n[Text Search]")
text_query = "visible spiral arms"
print(f"Query: '{text_query}'")

client = OpenAI()
response = client.embeddings.create(input=[text_query], model="text-embedding-3-large")
text_embedding = torch.tensor([response.data[0].embedding], device=device)
text_proj = clip_model.text_projector(text_embedding).cpu().numpy().flatten()

similarities = all_embeddings @ text_proj
best_idx = np.argmax(similarities)
match = train_data[best_idx]

print(f"\nBest match: RA={match['ra']:.4f}, Dec={match['dec']:.4f}")
print(f"  Spiral arms: {match['has-spiral-arms_yes']}/{match['has-spiral-arms_total-votes']} human votes, Cosine similarity: {similarities[best_idx]:.4f}")
print(f"  https://www.legacysurvey.org/viewer/cutout.jpg?ra={match['ra']}&dec={match['dec']}&layer=ls-dr10&pixscale=0.262")
