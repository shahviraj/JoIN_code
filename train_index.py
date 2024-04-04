'''
Trial function:
Build the index of the random samples in the latent space for calculating the nearest neighbors for kNN loss
'''
import numpy as np
import faiss
import sys
import os
import time

index_path = sys.argv[1][:-4]+'.bin'
if os.path.exists(index_path):
    raise ValueError("Index file already exists!")

clip_cache = np.load(sys.argv[1])

sttime = time.time()
print(f"Building Index...")
# Create a new index
index = faiss.index_factory(clip_cache.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)

# Add the vectors to the index
index.add(clip_cache)

print(f"Index built. total time taken: {(time.time() - sttime) / 60} minutes. Saving it...")
# Write the index to disk

faiss.write_index(index, index_path)

print("Done!")