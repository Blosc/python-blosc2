from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for concurrent execution
from time import time
import numba as nb
import numexpr
import numpy as np
from matplotlib import pyplot as plt
import blosc2  # Import Blosc2 library for high-performance compression

# Define a function named mean_numba decorated with @nb.jit to compile the code with Numba
# nopython=True and parallel=True specify that nopython mode will be used and
# the loop will attempt to be parallelized with Numba
def run_numba(num_threads, a, b, c):
    nb.set_num_threads(num_threads)  # Set the number of threads for Numba
    @nb.jit(nopython=True, parallel=True, nogil=True)
    def mean_numba(a, b, c):
        out = np.empty(a.shape, a.dtype)  # Create an empty array to store the result
        for i in nb.prange(a.shape[0]):  # Iterate over the first dimension of array a in parallel
            for j in nb.prange(a.shape[1]):  # Iterate over the second dimension of array a in parallel
                out[i, j] = (a[i, j] + b[i, j] * c[i, j]) + 2  # Calculate the corresponding value in the resulting array
        return out  # Return the resulting array

    # Measure the execution time of the approach with Numba
    start = time()  # Initial time
    mean_expr = mean_numba(a, b, c)  # Call the mean_numba function
    t = time() - start  # Calculate elapsed time
    return t

def run_lazy_expr(num_threads, a, b, c, cparams):
    blosc2.set_nthreads(num_threads)  # Set the number of threads for Blosc2 compression
    numexpr.set_num_threads(num_threads)  # Set the number of threads for NumExpr evaluation
    # Convert NumPy arrays to compressed arrays using Blosc2
    a1 = blosc2.asarray(a, cparams=cparams)  # Compressed array a
    b1 = blosc2.asarray(b, cparams=cparams)  # Compressed array b
    c1 = blosc2.asarray(c, cparams=cparams)  # Compressed array c
    # Perform the mathematical operation using LazyExpr: addition, multiplication, and addition of 2
    expr = a1 + b1 * c1  # LazyExpr expression
    expr += 2  # Add 2
    t = time()  # Initial time
    res = expr.evaluate(cparams=cparams)  # Evaluate the LazyExpr expression and get the result
    tt = time() - t  # Calculate elapsed time
    return tt

# Main program
# Create NumPy arrays a, b, and c with size 3000x4000 and values generated using linspace
dtype = np.float64  # Data type for the arrays
reshape = [30000, 4000]  # Reshape for arrays
size = reshape[0] * reshape[1]   # Total size of the arrays (30000 * 4000) = 120_000_000
a = np.linspace(0, 10, num=size, dtype=dtype).reshape(reshape)  # Array a
b = np.linspace(0, 10, num=size, dtype=dtype).reshape(reshape)  # Array b
c = np.linspace(0, 10, num=size, dtype=dtype).reshape(reshape)  # Array c
mega_bytes = (size // 8) // 1024 ** 2  # Total megabytes
threads = [1, 2, 3, 4]  # Number of threads
numba_times = []  # List to store Numba execution times
lazy_expr_times = []  # List to store LazyExpr execution times
clevels = [0, 1, 5]  # Compression levels
mean_expr = run_numba(1, a, b, c)  # Call run_numba to warm up Numba and ensure accurate timings
speed = []  # List to store speeds for plotting

# Calculate execution times for Numba
for num_thread in threads:
    numba = mega_bytes / run_numba(num_thread, a, b, c)
    numba_times.append(numba)
    speed.append(numba)

# Calculate execution times for LazyExpr
for clevel in clevels:
    lazy_expr_thread_times = []
    for num_thread in threads:
        cparams = {"clevel": clevel, "codec": blosc2.Codec.LZ4}  # Compression parameters
        lazy = mega_bytes / run_lazy_expr(num_thread, a, b, c, cparams)
        lazy_expr_thread_times.append(lazy)
        speed.append(lazy)
    lazy_expr_times.append(lazy_expr_thread_times)

# Graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots
limit_y = max(speed) + 1  # Limit for y-axis

# Graph Numba
ax1.plot(threads, numba_times, marker='o', label='Numba')  # Plot Numba data
ax1.set_xlabel('Number of threads')  # Set x-axis label
ax1.set_ylabel('Speed (MB/s)')  # Set y-axis label
ax1.set_title('Execution Time (Numba) vs. Number of threads')  # Set title
ax1.legend()  # Add legend
ax1.set_ylim(0, limit_y)  # Set y-axis limit

# Graph LazyExpr
for i, clevel in enumerate(clevels):
    ax2.plot(threads, lazy_expr_times[i], marker='o', label=f"clevel={clevel}")  # Plot LazyExpr data
ax2.set_xlabel('Number of threads')  # Set x-axis label
ax2.set_ylabel('Speed (MB/s)')  # Set y-axis label
ax2.set_title('LazyExpr')  # Set title
ax2.legend()  # Add legend
ax2.set_ylim(0, limit_y)  # Set y-axis limit
plt.show()  # Show the plot
