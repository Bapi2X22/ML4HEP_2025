import numpy as np

def poisson_distribution(mu, b, n):
    from scipy.stats import poisson
    return poisson.pmf(n, mu + b)

def get_range(mu, b=3.0, max_n=40, CL=0.9, verbose=False, log_file="intermediate_steps.txt"):
    # Step 1: Calculate P_1
    P_1_list = np.array([poisson_distribution(mu, b, n) for n in range(max_n)])
    
    # Step 2: Calculate P_2
    P_2_list = np.array([
        poisson_distribution(max(0, n - b), b, n) for n in range(max_n)
    ])

    # Step 3: Profile likelihood ratio
    Profile_likelihood_ratio = P_1_list / P_2_list

    # Step 4: Sort
    sorted_indices = np.argsort(-Profile_likelihood_ratio)

    # Step 5: Sorted probabilities
    sorted_probs = P_1_list[sorted_indices]

    # Step 6: Cumulative sum
    cumulative_sum = np.cumsum(sorted_probs)

    # Step 7: Get cutoff index
    last_index = np.argmin(np.abs(cumulative_sum - CL))
    accepted_n = sorted_indices[:last_index + 1]
    n_min = np.min(accepted_n)
    n_max = np.max(accepted_n)

    if verbose:
        with open(log_file, "w") as f:
            f.write(f"mu = {mu}, b = {b}, max_n = {max_n}, CL = {CL}\n\n")
            f.write("P_1_list:\n")
            f.write(", ".join(f"{p:.5f}" for p in P_1_list) + "\n\n")

            f.write("P_2_list:\n")
            f.write(", ".join(f"{p:.5f}" for p in P_2_list) + "\n\n")

            f.write("Profile_likelihood_ratio:\n")
            f.write(", ".join(f"{r:.5f}" for r in Profile_likelihood_ratio) + "\n\n")

            f.write("Sorted indices (by descending PLR):\n")
            f.write(", ".join(str(i) for i in sorted_indices) + "\n\n")

            f.write("Sorted P_1 probabilities:\n")
            f.write(", ".join(f"{p:.5f}" for p in sorted_probs) + "\n\n")

            f.write("Cumulative sum of sorted P_1:\n")
            f.write(", ".join(f"{c:.5f}" for c in cumulative_sum) + "\n\n")
            f.write(f"Last index (cutoff to reach CL): {last_index}\n")
            f.write(f"Accepted n values: {accepted_n.tolist()}\n")
            f.write(f"n_min = {n_min}, n_max = {n_max}\n")

    return n_min, n_max


get_range(2.0, b=3.0, max_n=40, CL=0.9, verbose=True, log_file="intermediate_steps.txt")