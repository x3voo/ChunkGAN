import numpy as np
import math
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from N_mc2data import MCReader, Block
import ChunkGAN

from itertools import product
from typing import Dict, List, Union
from Levenshtein import distance as levenshtein_distance
#python_Levenshtein==0.12.2


def compute_levenshtein(real: np.ndarray, generated: List[np.ndarray]):
    generated_str = ["".join(gen.flatten().astype(str)) for gen in generated]
    distances = [levenshtein_distance(gen_str_1, gen_str_2)
                 for gen_str_1, gen_str_2 in product(generated_str, generated_str)]
    print(distances)
    return np.mean(distances), np.var(distances)



def compute_prob(pattern_count, num_patterns, epsilon=1e-7):
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))

def pattern_key(level_slice):
    key = ""
    for line in level_slice:
        for token in line:
            key += str(token)
    return key

def get_pattern_counts(level, pattern_size):
    pattern_counts = defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            for inside in range(level.shape[2] - pattern_size + 1):
                down = up + pattern_size
                right = left + pattern_size
                outside = inside + pattern_size
                level_slice = level[up:down, left:right, inside:outside]
                pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts

def compute_pattern_counts(levels, pattern_size):
    """Compute pattern counts for multiple levels in parallel."""
    print(f"[{pattern_size}] Get pattern counts")
    with Pool() as pool:
        counts_per_level = pool.map(partial(get_pattern_counts, pattern_size=pattern_size), levels)
    #counts_per_level = []

    #for level in levels:
    #    pattern_counts_temp = get_pattern_counts(level, pattern_size)
    #    counts_per_level.append(pattern_counts_temp)
    '''
    pattern_counts = defaultdict(int)
    print(f"[{pattern_size}] Counting")
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    '''
    return counts_per_level

def compute_tpkldiv(real, generated, pattern_sizes, weight=0.5):
    dists = defaultdict(list)
    
    for pattern_size in pattern_sizes:
        print(f"Computing TP KL-Div for patterns of size {pattern_size}")
        real_pattern_counts = compute_pattern_counts([real], pattern_size)[0]
        generated_pattern_counts_per_level = compute_pattern_counts(generated, pattern_size)
        
        num_patterns = sum(real_pattern_counts.values())

        for generated_pattern_counts in tqdm(generated_pattern_counts_per_level):
            num_test_patterns = sum(generated_pattern_counts.values())
            kl_divergence_pq = 0
            for pattern in real_pattern_counts.keys():
                prob_p = compute_prob(real_pattern_counts[pattern], num_patterns)
                prob_q = compute_prob(generated_pattern_counts[pattern], num_test_patterns)
                
                kl_divergence_pq += prob_p * math.log(prob_p / prob_q)
                
            kl_divergence_qp = 0
            for pattern in generated_pattern_counts.keys():
                prob_q = compute_prob(real_pattern_counts[pattern], num_patterns)
                prob_p = compute_prob(generated_pattern_counts[pattern], num_test_patterns)
                
                kl_divergence_qp += prob_p * math.log(prob_p / prob_q)
                
            kl_divergence = weight * kl_divergence_qp + (1 - weight) * kl_divergence_pq
            
        
            dists[pattern_size].append(kl_divergence)

    for k, v in dists.items():
        print(f"[{k}] {v}")
        
    mean_tpkldiv = {k: np.mean(v) for k, v in dists.items()}
    var_tpkldiv = {k: np.var(v) for k, v in dists.items()}
    
    return mean_tpkldiv, var_tpkldiv

# One sample
def gradient_difference(real, generated):
    grad_diff = 0

    # diagonal XZ-axies gradient
    for x in range(15): # out of bounds
        for z in range(15):
            # Euclidean distance
            gen_grad = math.sqrt(pow(generated[x + 1, z] - generated[x, z], 2) + pow(generated[x, z + 1] - generated[x, z], 2))
            real_grad = math.sqrt(pow(real[x + 1, z] - real[x, z], 2) + pow(real[x, z + 1] - real[x, z], 2))
            
            grad_diff += abs(gen_grad - real_grad)

            
    return grad_diff / (15*15)

# Mean Gradient Magnitude Difference (or Mean Gradient Difference), MGD
def compute_gradient_difference(real_maps, generated_maps):
    grad_diffs = []
    sample_idx = 1
    
    for real, gen in zip(real_maps, generated_maps):
        diff = gradient_difference(real, gen)
        grad_diffs.append(diff)
        print(f"{sample_idx}, {diff}")
        sample_idx += 1

    max_height = 320 # max height will also be the maximum "error"
    MGD = np.mean(grad_diffs)
    similarity = (1 - (MGD / 320)) * 100
    print(f"Similarity: {similarity} %")
    
    return MGD, np.var(grad_diffs)
    


def main():
    # Chunk coords
    x = 0
    z = 0
    
    data = MCReader(64)
    data.load("DATA123")
    real_chunk = data.chunks[x,z]

    model = "one-hot_no-cave-loss_full-dis-unet"
    ChunkGAN.build()
    ChunkGAN.load_model(model)

    samples = 16
    
    heightmaps = np.zeros((samples, 16, 16), dtype=np.uint8)
    heightmaps = [data.heightmaps[x,z]] * samples

    GD = True
    if GD:
        for d in range(samples):
            heightmaps[d] = data.heightmaps[x,d]
    
        
    cave_dens = np.reshape([data.cave_densities[x,z]] * samples, (samples, 1))
    
    predict_chunks, _ = ChunkGAN.generate(samples, heightmaps, cave_dens)

    generated_chunks = []
    for chunk in predict_chunks:
        generated_chunks.append(data.decode(chunk))

    print(np.shape(generated_chunks))

    #real_chunk = np.random.randint(0, 6, (320, 16, 16))  # Example real chunk
    #generated_chunks = [np.random.randint(0, 6, (320, 16, 16)) for _ in range(2)]  # Example generated chunks

    
    if GD:
        print(f"heightmaps: {np.max(heightmaps)} : {np.min(heightmaps)}")

        REAL_heightmaps = np.zeros((samples, 16, 16), dtype=np.float32)
        GEN_heightmaps = np.zeros((samples, 16, 16), dtype=np.float32)
        
        for d in range(samples):
            REAL_heightmaps[d] = (heightmaps[d] + 64).astype(np.float32)
            GEN_heightmaps[d] = (data.create_heightmap(generated_chunks[d]) + 64).astype(np.float32)

        print(f"REAL_heightmaps: {np.max(REAL_heightmaps)} : {np.min(REAL_heightmaps)}")
        print(f"GEN_heightmaps: {np.max(GEN_heightmaps)} : {np.min(GEN_heightmaps)}")

        print(REAL_heightmaps[0])
        print(GEN_heightmaps[0])

        mean_GD, var_GD = compute_gradient_difference(REAL_heightmaps, GEN_heightmaps)
        print(f"Mean Gradient Difference: {mean_GD}")
        print(f"Variance Gradient Difference: {var_GD}")
        return 0


    mean_levenshtein, var_levenshtein = compute_levenshtein(real_chunk, generated_chunks)
    print(f"Mean levenshtein: {mean_levenshtein}")
    print(f"Variance levenshtein: {var_levenshtein}")
    #return 0
    
    pattern_sizes = [5, 10]

    mean_tpkldiv, var_tpkldiv = compute_tpkldiv(real_chunk, generated_chunks, pattern_sizes)
    print(f"Mean TPKL-Div: {mean_tpkldiv}")
    print(f"Variance TPKL-Div: {var_tpkldiv}")

if __name__ == "__main__":
    # This ensures that Windows correctly handles multiprocessing
    #mp.freeze_support()  # This is often not necessary unless you're creating an executable, but it can be helpful
    
    main()  # Call your main function
