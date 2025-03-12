#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Plots for the jit vs. numpy benchmarks on different array sizes and platforms.

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

iobw = True  # use I/O bandwidth instead of time

sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
#sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70]
sizes_GB = np.array([n * 1000 * n * 1000 * 8 * 2 / 2**30 for n in sizes])

amd = True

# Default title
title_ = "np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)"

# Load the data
if amd:
    #title_ = "AMD Ryzen 9 9800X3D (64 GB RAM)"

    create_l0 = [ 0.0325, 0.2709, 1.0339, 4.0489, 9.0849, 12.4154, 16.7818, 25.5946, 47.5691, 35.9919, 45.4295, 93.3075, 66.6529 ]
    compute_l0 = [ 0.0017, 0.0243, 0.0869, 0.3370, 0.7665, 1.0375, 1.3727, 1.7377, 2.1472, 2.6205, 3.0435, 18.5878, 28.0816 ]

    create_l0_dask = [ 0.0069, 0.0732, 0.2795, 1.2008, 2.7573, 4.9718, 7.2144, 32.7518, 113.6138, 160.8212, 197.4543, 218.0104, 236.6929 ]
    compute_l0_dask = [ 0.0166, 0.1251, 0.4104, 1.6123, 3.7044, 5.6765, 8.1201, 14.9497, 13.0838, 16.0741, 19.0059, 26.8003, 28.9472 ]

    create_l0_disk = [ 0.0305, 0.3371, 1.3249, 5.0602, 11.0410, 16.3685, 22.2012, 27.1348, 31.7409, 38.0690, 47.4424, 56.9335, 62.6965, 65.2226, 81.1631, 92.8310, 103.7345, 112.1973, 124.5319 ]
    compute_l0_disk = [ 0.0019, 0.0243, 0.0885, 0.3434, 0.7761, 1.0724, 1.4082, 1.7373, 2.1827, 2.6124, 7.0940, 9.0734, 10.1089, 11.2911, 13.0464, 22.6369, 25.4538, 28.7107, 31.9562 ]

    create_BLOSCLZ_l7 = [ 0.0267, 0.2610, 1.0299, 3.9724, 9.1326, 11.7598, 16.0252, 20.1420, 24.7293, 33.8753, 37.2400, 41.9200, 48.4979, 53.1935, 61.3910, 70.3354, 79.8628, 84.3074, 95.8080, 107.0405, 117.4525 ]
    compute_BLOSCLZ_l7 = [ 0.0018, 0.0205, 0.0773, 0.2931, 0.6938, 0.9001, 1.1693, 1.4701, 1.8559, 3.3739, 2.7486, 3.2836, 3.5230, 4.1417, 4.8597, 5.5748, 5.9453, 6.9264, 7.3589, 8.3207, 9.1710 ]

    create_BLOSCLZ_l7_disk = [ 0.0701, 0.2656, 1.0553, 4.0486, 9.2255, 12.2674, 16.4618, 20.1527, 25.3657, 33.7537, 37.3551, 43.0586, 48.4968, 53.9183, 62.9415, 71.7656, 80.5597, 85.5704, 97.0770, 109.7463, 119.2675 ]
    compute_BLOSCLZ_l7_disk = [ 0.0019, 0.0213, 0.0788, 0.3002, 0.7252, 0.9276, 1.2053, 1.4999, 1.9109, 3.4081, 2.8205, 3.3593, 3.6086, 4.2295, 4.9548, 5.6996, 6.0085, 7.0802, 7.4786, 8.4466, 9.4861 ]

    create_LZ4_l1 = [ 0.0304, 0.2582, 1.0298, 3.9502, 8.8945, 11.9267, 16.3965, 20.2368, 24.6837, 29.3425, 36.2631, 42.1709, 48.0605, 52.3962, 61.5175, 68.6328, 80.1160, 85.4322, 97.1122, 106.9973, 114.8584 ]
    compute_LZ4_l1 = [ 0.0018, 0.0210, 0.0756, 0.3003, 0.6609, 0.8886, 1.1285, 1.4453, 1.7959, 2.1889, 2.6978, 3.1586, 3.4286, 3.9929, 4.4590, 5.3601, 5.6702, 6.4690, 6.9764, 7.8714, 8.6404 ]

    create_LZ4_l1_dask = [ 0.0071, 0.0800, 0.2855, 1.1456, 2.6405, 3.4453, 20.8665, 25.2932, 53.7019, 68.1571, 98.4894, 175.2592, 197.0002 ]
    compute_LZ4_l1_dask = [ 0.0162, 0.1174, 0.4152, 1.5343, 3.5179, 4.7557, 7.7030, 8.8297, 12.0453, 14.0156, 17.0496, 18.7882, 21.5925 ]

    create_LZ4_l1_disk = [ 1.7980, 0.2617, 1.0480, 4.0809, 9.0720, 13.8294, 16.7269, 20.5108, 24.9465, 30.0428, 37.1903, 42.8075, 48.7775, 52.9890, 63.4071, 70.1766, 81.9747, 88.1830, 97.7921, 111.0611, 119.7673 ]
    compute_LZ4_l1_disk = [ 0.0019, 0.0214, 0.0795, 0.3060, 0.6985, 0.9195, 1.1766, 1.5213, 1.8845, 2.2972, 2.8044, 3.2587, 3.5898, 4.1524, 4.6293, 5.5485, 5.8715, 6.7386, 7.3019, 8.2307, 9.0145 ]

    create_ZSTD_l1 = [ 0.0302, 0.2704, 1.0703, 4.1243, 9.2185, 12.5026, 17.0585, 20.8708, 25.5844, 31.0571, 37.7114, 42.8297, 50.2696, 54.5773, 63.6311, 73.0370, 84.0092, 89.0686, 100.3300, 108.8173, 119.1154 ]
    compute_ZSTD_l1 = [ 0.0021, 0.0296, 0.1045, 0.3979, 0.8787, 1.3064, 1.7404, 2.1938, 2.6780, 3.3929, 3.8601, 4.3665, 5.0127, 5.7346, 6.1056, 7.9448, 8.2872, 9.4659, 9.2376, 10.4273, 11.6572 ]

    create_ZSTD_l1_dask = [ 0.0079, 0.0872, 0.2974, 1.0849, 2.6028, 3.4071, 18.5250, 25.3142, 54.4772, 63.5289, 85.9178, 144.4604, 196.1394 ]
    compute_ZSTD_l1_dask = [ 0.0164, 0.1186, 0.4032, 1.5453, 3.4972, 4.7853, 7.6398, 8.5793, 12.1144, 14.1863, 17.8496, 19.0857, 21.8183 ]

    create_ZSTD_l1_disk = [ 0.6564, 0.2825, 1.0826, 4.1968, 9.5022, 13.4840, 17.5387, 21.5807, 26.0052, 31.3524, 38.5889, 44.1105, 49.8849, 55.5297, 64.6479, 72.7471, 84.6595, 90.4970, 99.9710, 111.6817, 120.8941 ]
    compute_ZSTD_l1_disk = [ 0.0022, 0.0300, 0.1066, 0.4099, 0.8974, 1.3218, 1.7679, 2.2154, 2.7007, 3.4267, 3.9255, 4.4597, 5.1155, 5.8251, 6.2064, 8.0141, 8.4316, 9.3195, 9.4570, 10.7034, 11.9192 ]

    create_numpy = [ 0.0020, 0.0527, 0.2292, 0.9412, 2.1043, 2.8286, 3.7046, 4.7217, 5.8308, 7.0491 ]
    compute_numpy = [ 0.0179, 0.2495, 0.9840, 3.9263, 8.8450, 12.0259, 16.3507, 40.1672, 155.1292, 302.5115 ]

    create_numpy_dask = [ 0.0007, 0.0378, 0.1640, 0.6665, 1.5046, 2.0726, 2.7750, 4.6960, 5.7110, 41.2241 ]
    compute_numpy_dask = [ 0.0169, 0.3955, 1.5680, 6.2638, 14.0860, 19.2658, 32.2012, 70.2960, 368.6261, 392.6483 ]

    create_numpy_numba = [ 0.0013, 0.0401, 0.1643, 0.6682, 1.5016, 2.0528, 2.6803, 3.4313, 5.5713, 15.3014, 23.5496, 43.5016, 62.5048 ]
    compute_numpy_numba = [ 0.0932, 0.0317, 0.1569, 0.7485, 1.9492, 2.8305, 3.8708, 5.2393, 6.8156, 8.3882, 12.2608, 25.4770, 37.2782 ]

    create_numpy_jit = [ 0.0019, 0.0529, 0.2261, 0.9219, 2.0589, 2.8350, 3.7131, 18.4375, 26.5959, 34.5221, 33.7157, 49.6762, 63.1401 ]
    compute_numpy_jit = [ 0.0035, 0.0180, 0.0622, 0.2307, 0.5196, 0.7095, 0.9251, 1.1981, 1.4729, 2.2007, 2.0953, 12.6746, 26.6424 ]


yaxis_title = 'Time (s)'
if iobw:
    yaxis_title = 'I/O bandwidth (GB/s)'
    # Convert times to I/O bandwidth
    create_l0 = sizes_GB[:len(create_l0)] / np.array(create_l0)
    compute_l0 = sizes_GB[:len(compute_l0)] / np.array(compute_l0)
    create_l0_disk = sizes_GB[:len(create_l0_disk)] / np.array(create_l0_disk)
    compute_l0_disk = sizes_GB[:len(compute_l0_disk)] / np.array(compute_l0_disk)
    create_l0_dask = sizes_GB[:len(create_l0_dask)] / np.array(create_l0_dask)
    compute_l0_dask = sizes_GB[:len(compute_l0_dask)] / np.array(compute_l0_dask)
    create_BLOSCLZ_l7 = sizes_GB[:len(create_BLOSCLZ_l7)] / np.array(create_BLOSCLZ_l7)
    compute_BLOSCLZ_l7 = sizes_GB[:len(compute_BLOSCLZ_l7)] / np.array(compute_BLOSCLZ_l7)
    create_BLOSCLZ_l7_disk = sizes_GB[:len(create_BLOSCLZ_l7_disk)] / np.array(create_BLOSCLZ_l7_disk)
    compute_BLOSCLZ_l7_disk = sizes_GB[:len(compute_BLOSCLZ_l7_disk)] / np.array(compute_BLOSCLZ_l7_disk)
    create_LZ4_l1 = sizes_GB[:len(create_LZ4_l1)] / np.array(create_LZ4_l1)
    compute_LZ4_l1 = sizes_GB[:len(compute_LZ4_l1)] / np.array(compute_LZ4_l1)
    create_LZ4_l1_disk = sizes_GB[:len(create_LZ4_l1_disk)] / np.array(create_LZ4_l1_disk)
    compute_LZ4_l1_disk = sizes_GB[:len(compute_LZ4_l1_disk)] / np.array(compute_LZ4_l1_disk)
    create_LZ4_l1_dask = sizes_GB[:len(create_LZ4_l1_dask)] / np.array(create_LZ4_l1_dask)
    compute_LZ4_l1_dask = sizes_GB[:len(compute_LZ4_l1_dask)] / np.array(compute_LZ4_l1_dask)
    create_ZSTD_l1 = sizes_GB[:len(create_ZSTD_l1)] / np.array(create_ZSTD_l1)
    compute_ZSTD_l1 = sizes_GB[:len(compute_ZSTD_l1)] / np.array(compute_ZSTD_l1)
    create_ZSTD_l1_disk = sizes_GB[:len(create_ZSTD_l1_disk)] / np.array(create_ZSTD_l1_disk)
    compute_ZSTD_l1_disk = sizes_GB[:len(compute_ZSTD_l1_disk)] / np.array(compute_ZSTD_l1_disk)
    create_ZSTD_l1_dask = sizes_GB[:len(create_ZSTD_l1_dask)] / np.array(create_ZSTD_l1_dask)
    compute_ZSTD_l1_dask = sizes_GB[:len(compute_ZSTD_l1_dask)] / np.array(compute_ZSTD_l1_dask)
    create_numpy = sizes_GB[:len(create_numpy)] / np.array(create_numpy)
    compute_numpy = sizes_GB[:len(compute_numpy)] / np.array(compute_numpy)
    create_numpy_dask = sizes_GB[:len(create_numpy_dask)] / np.array(create_numpy_dask)
    compute_numpy_dask = sizes_GB[:len(compute_numpy_dask)] / np.array(compute_numpy_dask)
    create_numpy_numba = sizes_GB[:len(create_numpy_numba)] / np.array(create_numpy_numba)
    compute_numpy_numba = sizes_GB[:len(compute_numpy_numba)] / np.array(compute_numpy_numba)
    create_numpy_jit = sizes_GB[:len(create_numpy_jit)] / np.array(create_numpy_jit)
    compute_numpy_jit = sizes_GB[:len(compute_numpy_jit)] / np.array(compute_numpy_jit)

def add_ram_limit(figure, compute=True):
    y1_max = 25 if compute else 2
    if amd:
        #y1_max = 35 if compute else y1_max
        figure.add_shape(
            type="line", x0=64, y0=0, x1=64, y1=y1_max,
            line=dict(color="Gray", width=2, dash="dot"),
        )
        figure.add_annotation(x=64, y=y1_max * .9, text="64 GB RAM", showarrow=True, arrowhead=2, ax=45, ay=0)

# Plot the data. There will be 2 plots: one for create times and another for compute times
labels = dict(
    l0="Blosc2 + NDArray (No compression)",
    l0_dask="Dask + Zarr (No compression)",
    LZ4_l1="Blosc2 + NDArray (LZ4, lvl=1)",
    LZ4_l1_dask="Dask + Zarr (Blosc+LZ4, lvl=1)",
    ZSTD_l1="Blosc2 (ZSTD, lvl=1)",
    ZSTD_l1_dask="Dask + Zarr (Blosc+ZSTD, lvl=1)",
    numpy="NumPy",
    numpy_jit="Blosc2 + NumPy",
    numpy_dask="Dask + NumPy",
    numpy_numba="Numba + NumPy",
)

# Create the create times plot
fig_create = go.Figure()
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_l0, mode='lines+markers', name=labels["l0"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_l0_dask, mode='lines+markers', name=labels["l0_dask"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_LZ4_l1, mode='lines+markers', name=labels["LZ4_l1"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_LZ4_l1_dask, mode='lines+markers', name=labels["LZ4_l1_dask"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l1, mode='lines+markers', name=labels["ZSTD_l1"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l1_dask, mode='lines+markers', name=labels["ZSTD_l1_dask"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_numpy_numba, mode='lines+markers',
               name=labels["numpy_numba"], line=dict(color='black', dash='dot')))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_numpy_jit, mode='lines+markers',
               name=labels["numpy"], line=dict(color='brown')))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_numpy_jit, mode='lines+markers',
               name=labels["numpy_dask"], line=dict(color='cyan')))
fig_create.update_layout(title=f'Create operands: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title)

# Add a vertical line at RAM limit
add_ram_limit(fig_create, compute=False)

# Create the compute times plot
# Calculate the maximum y1 value
y1_max = max(max(compute_l0), max(compute_l0_disk), max(compute_LZ4_l1), max(compute_LZ4_l1_disk),
             max(compute_ZSTD_l1), max(compute_ZSTD_l1_disk), max(compute_numpy), max(compute_numpy_jit),
             max(compute_numpy_numba))

fig_compute = go.Figure()
# fig_compute.add_trace(
#     go.Scatter(x=sizes_GB, y=compute_numpy_jit, mode='lines+markers', name=labels["numpy_jit"], line=dict(color='brown', dash='dot')))
# fig_compute.add_trace(
#     go.Scatter(x=sizes_GB, y=compute_numpy_dask, mode='lines+markers', name=labels["numpy_dask"], line=dict(color='orange', dash='dot')))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_l0, mode='lines+markers', name=labels["l0"], line=dict(color='blue')))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_LZ4_l1[:15], mode='lines+markers', name=labels["LZ4_l1"], line=dict(color='green')))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_l0_dask, mode='lines+markers', name=labels["l0_dask"], line=dict(color='red', dash='dash')))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_LZ4_l1_dask, mode='lines+markers', name=labels["LZ4_l1_dask"], line=dict(color='purple', dash='dash')))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_numpy_numba, mode='lines+markers', name=labels["numpy_numba"], line=dict(color='black', dash='dot')))
fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy, mode='lines+markers',
                                 name=labels["numpy"], line=dict(color='grey', dash='dot')))
fig_compute.update_layout(title=f'Blosc2 vs others compute: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title)

# Add a vertical line at RAM limit
add_ram_limit(fig_compute, compute=True)

# Show the plots
fig_create.show()
fig_compute.show()
