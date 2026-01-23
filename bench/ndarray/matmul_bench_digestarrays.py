#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# It is important to force numpy to use mkl as it can speed up the
# blosc2 matmul (which uses np.matmul as a backend) by a factor of 2x:
# conda install numpy mkl

import numpy as np
import blosc2
import time
import matplotlib.pyplot as plt
import torch
import pickle


plt.rcParams.update({'text.usetex':False,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rc('text', usetex=False)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')

ndim = 3
filename = f"matmul{ndim}D_bench"

shapes = np.array([1, 2, 4, 8, 12, 16, 20])**(1/3) * 2**(28/3)
plotmode = True
if not plotmode:
    for xp in [blosc2, np, torch]:
        sizes = []
        mean_times = {'blosc2':[], 'torch':[], 'numpy':[]}
        for n in shapes:
            N = int(n)
            shape_a = (N,) * ndim
            shape_b = (N,) * ndim
            size_gb = (N ** ndim * 4) / (2 ** 30)

            for lib in [blosc2, torch, np]:
                # Generate matrices
                matrix_a = lib.full(shape_a, fill_value=3., dtype=lib.float32)
                matrix_b = lib.full(shape_b, fill_value=2.4, dtype=lib.float32)
                matrix_c = lib.full(shape_b[:1], fill_value=.4, dtype=lib.float32)
                _time = 0
                #multiplication
                if (xp.__name__ == 'torch' and lib.__name__ == 'torch'
                    ) or (xp.__name__ == 'numpy' and lib.__name__ != 'blosc2'
                        ) or xp.__name__ == 'blosc2':
                    for _ in range(1):
                        t0 = time.perf_counter()
                        if xp.__name__ == 'blosc2':
                            (xp.matmul(matrix_a, matrix_b) + matrix_c).compute()
                        else:
                            xp.matmul(matrix_a, matrix_b) + matrix_c
                        _time = time.perf_counter() - t0
                mean_times[lib.__name__]+=[_time]
                print(f"Size = {np.round(size_gb, 1)} GB, {xp.__name__.upper()}_{lib.__name__} Performance = {_time:.2f} s")

            sizes+=[size_gb * 3]

        with open(f"{filename}_{xp.__name__.upper()}.pkl", 'wb') as f:
            pickle.dump(
                {'blosc2':{
            "Matrix Size (GB)": sizes,
            "Mean Time (s)": mean_times['blosc2']
            },
        'numpy':{
            "Matrix Size (GB)": sizes,
            "Mean Time (s)": mean_times['numpy']
        },
        'torch':{
            "Matrix Size (GB)": sizes,
            "Mean Time (s)": mean_times['torch']
        }
        }, f)

else:
    plt.figure()
    for mkr, xp in zip(('X', 'd', 's'), [blosc2, torch, np]):
        with open(f"{filename}_{xp.__name__.upper()}.pkl", 'rb') as f:
            res_dict = pickle.load(f)

        # Create plots for Numpy vs Blosc vs Torch
        _dict = res_dict['torch']
        x=np.round(_dict["Matrix Size (GB)"], 1)
        plt.plot(x, _dict["Mean Time (s)"], color='r', label=f'{xp.__name__.upper()}_torch', marker = mkr)
        if xp.__name__ != 'torch':
            _dict = res_dict['numpy']
            plt.plot(x, _dict["Mean Time (s)"], color='g', label=f'{xp.__name__.upper()}_numpy', marker = mkr)
        if xp.__name__ == 'blosc2':
            _dict = res_dict['blosc2']
            plt.plot(x, _dict["Mean Time (s)"], color='b', label=f'{xp.__name__.upper()}_blosc2', marker = mkr)


    plt.xlabel('Working set size (GB)')
    plt.legend()
    plt.ylabel("Time (s)")
    plt.title(f'matmul(A, B) + c, ndim = {ndim}')
    plt.gca().set_yscale('log')
    plt.savefig(f'{filename}.png', format="png")
