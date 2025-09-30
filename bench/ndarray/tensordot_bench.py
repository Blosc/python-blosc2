import numpy as np
import blosc2
import time
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex':False,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rc('text', usetex=False)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')

filename = f"tensordot_bench"
width = 0.2
w = -width

shapes = [813, 931, 1024, 1103, 1173, 1291]
cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)

err_plus = []
err_minus = []
sizes = []
np_or_blosc2 = []
mean_times = []

for N in shapes:
    shape_a = (N,) * 3
    shape_b = (N,) * 3
    size_gb = (N * N * N * 8) / (2 ** 30)

    # Generate matrices
    matrix_a_blosc2 = blosc2.ones(shape=shape_a, cparams=cparams, chunks=(140,)*3)
    matrix_b_blosc2 = blosc2.ones(shape=shape_b, cparams=cparams, chunks=(140,)*3)
    matrix_a_np = matrix_a_blosc2[:]
    matrix_b_np = matrix_b_blosc2[:]
    blosc_mean, blosc_max, blosc_min = 0, -np.inf, np.inf
    np_mean, np_max, np_min = 0, -np.inf, np.inf

    for axis in ((0, 1), (1, 2), (2, 0)):
        # Blosc2 multiplication
        t0 = time.perf_counter()
        result_blosc2 = blosc2.tensordot(matrix_a_blosc2, matrix_b_blosc2, axes=(axis, axis))
        blosc2_time = time.perf_counter() - t0

        # Compute GFLOPS
        blosc_mean += blosc2_time/3
        blosc_min = min(blosc_min, blosc2_time)
        blosc_max = max(blosc_max, blosc2_time)

        print(f"N, axes={N, axis}, Blosc2 Performance = {blosc2_time:.2f} s")

        # Numpy multiplication
        t0 = time.perf_counter()
        result_numpy = np.tensordot(matrix_a_np, matrix_b_np, axes=(axis, axis))
        numpy_time = time.perf_counter() - t0

        np_mean += numpy_time / 3
        np_min = min(np_min, numpy_time)
        np_max = max(np_max, numpy_time)

        print(f"N, axes={N, axis}, Numpy Performance = {numpy_time:.2f} s")
    sizes+=[size_gb, size_gb]
    err_minus+=[blosc_mean-blosc_min, np_mean-np_min]
    err_plus+=[blosc_max-blosc_mean, np_max-np_mean]
    mean_times+=[blosc_mean, np_mean]
    np_or_blosc2+=["Blosc2", "NumPy"]

import pickle
with open("tensordot_bench.pkl", 'wb') as f:
      pickle.dump(
           {'Blosc2':{
    "Matrix Size (GB)": sizes[::2],
    "Mean Time (s)": mean_times[::2],
    "Min time": err_minus[::2],
    "Max time": err_minus[::2],
    "Lib": np_or_blosc2[::2]
},
'NumPy':{
    "Matrix Size (GB)": sizes[1::2],
    "Mean Time (s)": mean_times[1::2],
    "Min time": err_minus[1::2],
    "Max time": err_minus[1::2],
    "Lib": np_or_blosc2[1::2]
}
}, f)

with open("tensordot_bench.pkl", 'rb') as f:
    res_dict = pickle.load(f)

# Create barplot for Numpy vs Blosc
blosc2_dict = res_dict['Blosc2']
x=np.arange(len(blosc2_dict["Matrix Size (GB)"]))
err = (blosc2_dict["Max time"], blosc2_dict["Min time"])
plt.bar(x + w, blosc2_dict["Mean Time (s)"], width, color='r', label='Blosc2', yerr=err, capsize=5, ecolor='k',
        error_kw=dict(lw=2, capthick=2, ecolor='k'))
w += width
numpy_dict = res_dict['NumPy']
err = (numpy_dict["Max time"], numpy_dict["Min time"])
plt.bar(x + w, numpy_dict["Mean Time (s)"], width, color='b', label='NumPy', yerr=err, capsize=5, ecolor='k',
        error_kw=dict(lw=2, capthick=2, ecolor='k'))

plt.xlabel('Array size (GB)')
plt.legend()
plt.xticks(x-width, np.round(blosc2_dict["Matrix Size (GB)"], 0))
plt.ylabel("Time (s)")
plt.title(f"Tensordot comparison, Blosc2 vs. Numpy (different axes sums)")
plt.gca().set_yscale('log')
plt.savefig(f'{filename}.png', format="png")
plt.show()

# Benchmark hypot
# import timeit
# import numpy as np
# import numexpr as ne

# # --- Experiment Setup ---
# n_frames = 20000  # Raise this for more frames
# dtype = np.float64  # Data type for the grid
# # --- Coordinate creation ---
# x = np.linspace(0, n_frames, n_frames, dtype=dtype)
# y = np.linspace(-4 * np.pi, 4 * np.pi, n_frames, dtype=dtype)
# X = np.expand_dims(x, (1, 2))  # Shape: (N, 1, 1)
# Y = np.expand_dims(x, (0, 2))  # Shape: (1, N, 1)

# print(f"Average time for np.hypot(X, Y): {timeit.timeit('np.hypot(X, Y)', globals=globals(), number=10)/10} s")
# print("Average time for ne.evaluate('hypot(X, Y)'): {0} s".format(timeit.timeit('ne.evaluate("hypot(X, Y)")', globals=globals(), number=10)/10))
# import blosc2
# print("Average time for blosc2.hypot(X, Y): {0} s".format(timeit.timeit('blosc2.hypot(X, Y).compute()', globals=globals(), number=10)/10))
