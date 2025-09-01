#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Example showing how lazy expressions can be used to quickly walk through
# a 3D array and visualize it with matplotlib. This example uses Blosc2
# arrays, but it can use NumPy arrays for comparison.

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

import blosc2

# --- Experiment Setup ---
scale = 1.0  # Scale factor for the grid
width, height = np.array((1000, 1000)) * scale  # Size of the grid
n_frames = int(1000 * scale)  # Raise this for more frames
dtype = np.float64  # Data type for the grid
use_blosc2 = True  # Set to False to use NumPy arrays instead
realize_blosc2 = False  # Set to False to skip Blosc2 realization
make_animation = True  # Set to False to skip animation creation
travel_dim = 2  # Dimension to travel through (0 for X, 1 for Y, 2 for Z)

# --- Coordinate creation ---
x = blosc2.linspace(0, n_frames, n_frames, dtype=dtype)
y = blosc2.linspace(-4 * np.pi, 4 * np.pi, width, dtype=dtype)
z = blosc2.linspace(-4 * np.pi, 4 * np.pi, height, dtype=dtype)
X = blosc2.expand_dims(x, (1, 2))  # Shape: (N, 1, 1)
Y = blosc2.expand_dims(y, (0, 2))  # Shape: (1, N, 1)
Z = blosc2.expand_dims(z, (0, 1))  # Shape: (1, 1, N)
if not use_blosc2:
    # If not using Blosc2, convert to NumPy arrays
    # X, Y, Z = np.meshgrid(x, y, z)
    X, Y, Z = X[:], Y[:], Z[:]  # more memory efficient

# Actual 3D function


# --- Helper Functions ---
def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# --- 3D Data Generation ---
def compute_3Ddata():
    time_factor = X * Y * 0.001
    R = np.sqrt(Y**2 + Z**2)
    theta = np.arctan2(Z, Y)
    return np.sin(R * 3 - time_factor * 2) * np.cos(theta * 3)


# --- Pre-computation ---
print("Generating frames...")
mem_before = get_memory_mb()
t0 = time.time()
frames = compute_3Ddata()
if realize_blosc2:
    frames = frames[:]
time_gen_frames = time.time() - t0
print(f"Frames generated in {time_gen_frames:.2f} seconds")
print(f"Memory used for frames: {get_memory_mb() - mem_before:.1f} MB")
print(f"Type of frames: {type(frames)}, dtype: {frames.dtype}")
print("Shape of frames:", frames.shape)


# --- Matplotlib Initial Frame ---
fig, ax = plt.subplots(figsize=(8, 8))
sl = (*(slice(None),) * travel_dim, 0)  # Select the slice for the travel dimension
im = ax.imshow(frames[sl], cmap="viridis")
fig.colorbar(im, ax=ax)
ax.set_title("Blosc2 Animated Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")


# --- Animation Update Function ---
start_time = time.time()


def update(frame_num):
    sl = (*(slice(None),) * travel_dim, frame_num)  # Select the slice for the travel dimension
    frame_array = frames[sl]
    # print(f"Type of frame_array: {type(frame_array)}, shape: {frame_array.shape}")
    # Evaluate the expression for the current frame on the fly
    im.set_array(frame_array)
    if frame_num < n_frames - 1:
        ax.set_title(f"Frame {frame_num + 1}/{n_frames}")
    else:
        # Final frame to show the total time
        elapsed_time = time.time() - start_time + time_gen_frames
        ax.set_title(f"Generated {n_frames} frames in {elapsed_time:.2f} seconds")
    return (im,)


# --- Matplotlib Animation ---
if make_animation:
    from matplotlib.animation import FuncAnimation

    mem_before = get_memory_mb()
    ani = FuncAnimation(fig, update, frames=n_frames, interval=10, blit=False, repeat=False)
    # To save as a file:
    # ani.save('blosc2_animation.gif', writer='imagemagick')
    print(f"Animation created, memory used: {get_memory_mb() - mem_before:.1f} MB")

# This loop is for performance testing and not required for the animation itself
mem_before = get_memory_mb()
t0 = time.time()
for i in range(n_frames):
    update(i)
print(f"Frames set to matplotlib in {time.time() - t0:.2f} seconds")
print(f"Memory used for matplotlib updates: {get_memory_mb() - mem_before:.1f} MB")

plt.show()
