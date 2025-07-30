#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

import blosc2

# --- Data Generation ---
width, height = np.array((1000, 1000)) * 1  # Size of the grid
n_frames = 1000
dtype = np.float64  # Data type for the grid
x = blosc2.linspace(0, n_frames, n_frames, dtype=dtype)
y = blosc2.linspace(-4 * np.pi, 4 * np.pi, width, dtype=dtype)
z = blosc2.linspace(-4 * np.pi, 4 * np.pi, height, dtype=dtype)
use_blosc2 = True  # Set to False to use NumPy arrays instead

X = blosc2.expand_dims(blosc2.expand_dims(x, 1), 2)  # Shape: (50, 1, 1)
Y = blosc2.expand_dims(blosc2.expand_dims(y, 0), 2)  # Shape: (1, 1024, 1)
Z = blosc2.expand_dims(blosc2.expand_dims(z, 0), 0)  # Shape: (1, 1, 1024)
if not use_blosc2:
    # X, Y, Z = np.meshgrid(x, y, z)
    X = X[:]
    Y = Y[:]
    Z = Z[:]


# --- Helper Functions ---


# @blosc2.jit(disable=use_blosc2)
def compute_3Ddata():
    """Generate a 3D dataset."""
    time_factor = X * 0.1
    R = np.sqrt(Y**2 + Z**2)
    theta = np.arctan2(Z, Y)
    return np.sin(R * 2 - time_factor * 2) * np.cos(theta * 3)


# For profiling memory usage
def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# --- Pre-computation ---
mem_before = get_memory_mb()
t0 = time.time()
# frames = compute_3Ddata().compute()
frames = compute_3Ddata()
print(f"Frames generated in {time.time() - t0:.2f} seconds")
print(f"Type of frames: {type(frames)}, dtype: {frames.dtype}")
print(f"Memory used for frames: {get_memory_mb() - mem_before:.1f} MB")
print("Shape of frames:", frames.shape)


# --- Matplotlib Animation ---
fig, ax = plt.subplots(figsize=(8, 8))

# Initial image - evaluate the first frame
initial_array = frames[0]
print(f"Type of initial_array: {type(initial_array)}, dtype: {initial_array.dtype}")
im = ax.imshow(initial_array, cmap="viridis")  # , interpolation='bicubic')
fig.colorbar(im, ax=ax)
ax.set_title("Blosc2 Animated Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

start_time = time.time()


def update(frame_num):
    """Update function for the animation."""
    if frame_num < n_frames:
        frame_array = frames[frame_num]
        # print(f"Type of frame_array: {type(frame_array)}, dtype: {frame_array.dtype}")
        # Evaluate the expression for the current frame on the fly
        im.set_array(frame_array)
        ax.set_title(f"Frame {frame_num + 1}/{n_frames}")
    else:
        # Final frame to show the total time
        elapsed_time = time.time() - start_time
        ax.set_title(f"Generated {n_frames} frames in {elapsed_time:.2f} seconds")
    return (im,)


# This loop is for performance testing and not required for the animation itself
mem_before = get_memory_mb()
t0 = time.time()
for i in range(n_frames + 1):
    update(i)
print(f"Frames set to matplotlib in {time.time() - t0:.2f} seconds")
print(f"Memory used for matplotlib updates: {get_memory_mb() - mem_before:.1f} MB")

# Create the animation
mem_before = get_memory_mb()
# This takes long time, so we comment it out and simulate the bulk update above
# ani = FuncAnimation(fig, update, frames=n_frames + 1, interval=10, blit=False, repeat=False)

# To display in a Jupyter Notebook, you would use:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save as a file:
# ani.save('blosc2_animation.gif', writer='imagemagick')

plt.show()
print(f"Animation created, memory used: {get_memory_mb() - mem_before:.1f} MB")
