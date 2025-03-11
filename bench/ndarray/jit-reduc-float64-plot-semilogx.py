#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Plots for the jit vs. numpy benchmarks on different array sizes and platforms.

import plotly.graph_objects as go
import numpy as np

iobw = True  # use I/O bandwidth instead of time

sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
         105, 110, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
sizes_GB = np.array([n * 1000 * n * 1000 * 8 * 2 / 2**30 for n in sizes])

# Default title
title_ = "np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)"

# Load the data
#title_ = "AMD Ryzen 9 9800X3D (64 GB RAM)"

create_l0 = [ 0.0325, 0.2709, 1.0339, 4.0489, 9.0849, 12.4154, 16.7818, 25.5946, 47.5691, 35.9919, 45.4295, 93.3075, 66.6529 ]
compute_l0 = [ 0.0017, 0.0243, 0.0869, 0.3370, 0.7665, 1.0375, 1.3727, 1.7377, 2.1472, 2.6205, 3.0435, 18.5878, 28.0816 ]

create_l0_disk = [ 0.0305, 0.3371, 1.3249, 5.0602, 11.0410, 16.3685, 22.2012, 27.1348, 31.7409, 38.0690, 47.4424, 56.9335, 62.6965, 65.2226, 81.1631, 92.8310, 103.7345, 112.1973, 124.5319 ]
compute_l0_disk = [ 0.0019, 0.0243, 0.0885, 0.3434, 0.7761, 1.0724, 1.4082, 1.7373, 2.1827, 2.6124, 7.0940, 9.0734, 10.1089, 11.2911, 13.0464, 22.6369, 25.4538, 28.7107, 31.9562 ]

create_LZ4_l1 = [ 0.0304, 0.2582, 1.0298, 3.9502, 8.8945, 11.9267, 16.3965, 20.2368, 24.6837, 29.3425, 36.2631, 42.1709, 48.0605, 52.3962, 61.5175, 68.6328, 80.1160, 85.4322, 97.1122, 106.9973, 114.8584, 219.8679, 372.3182, 650.2087, 876.6964, 1535.3019, 1717.6310, 2605.6513, 3490.7571, 4253.5521, 4192.6208, 6181.3742, 6793.9787, 7135.4944 ]
compute_LZ4_l1 = [ 0.0018, 0.0210, 0.0756, 0.3003, 0.6609, 0.8886, 1.1285, 1.4453, 1.7959, 2.1889, 2.6978, 3.1586, 3.4286, 3.9929, 4.4590, 5.3601, 5.6702, 6.4690, 6.9764, 7.8714, 8.6404, 15.7214, 29.4130, 46.5909, 87.1930, 164.6234, 258.9626, 256.4864, 378.0102, 476.1793, 585.9910, 734.2687, 853.7598, 727.2813 ]

create_LZ4_l1_disk = [ 1.7980, 0.2617, 1.0480, 4.0809, 9.0720, 13.8294, 16.7269, 20.5108, 24.9465, 30.0428, 37.1903, 42.8075, 48.7775, 52.9890, 63.4071, 70.1766, 81.9747, 88.1830, 97.7921, 111.0611, 119.7673, 214.8363, 370.7900, 600.6060, 872.7770, 1314.0561, 1581.3989, 1898.3007, 2910.3205, 3476.1479, 4753.6958, 5590.7596, 6627.1739, 6884.6506 ]
compute_LZ4_l1_disk = [ 0.0019, 0.0214, 0.0795, 0.3060, 0.6985, 0.9195, 1.1766, 1.5213, 1.8845, 2.2972, 2.8044, 3.2587, 3.5898, 4.1524, 4.6293, 5.5485, 5.8715, 6.7386, 7.3019, 8.2307, 9.0145, 16.1475, 30.1677, 59.1110, 81.9494, 112.0279, 169.0670, 173.9750, 248.5645, 332.5040, 354.8242, 448.8191, 493.8022, 570.6065 ]

create_ZSTD_l1 = [ 0.0302, 0.2704, 1.0703, 4.1243, 9.2185, 12.5026, 17.0585, 20.8708, 25.5844, 31.0571, 37.7114, 42.8297, 50.2696, 54.5773, 63.6311, 73.0370, 84.0092, 89.0686, 100.3300, 108.8173, 119.1154, 265.8825, 493.3042, 851.1048, 1165.6934, 1589.0762, 2055.2161, 2481.3166, 3501.0184, 4258.2440, 4151.9682, 6119.5858, 6518.2127, 7371.7506 ]
compute_ZSTD_l1 = [ 0.0021, 0.0296, 0.1045, 0.3979, 0.8787, 1.3064, 1.7404, 2.1938, 2.6780, 3.3929, 3.8601, 4.3665, 5.0127, 5.7346, 6.1056, 7.9448, 8.2872, 9.4659, 9.2376, 10.4273, 11.6572, 22.0410, 36.7011, 65.2484, 84.9773, 123.1597, 147.6101, 274.7479, 384.7447, 442.0842, 512.2530, 641.9793, 702.5878, 807.3979 ]

create_ZSTD_l1_disk = [ 0.6564, 0.2825, 1.0826, 4.1968, 9.5022, 13.4840, 17.5387, 21.5807, 26.0052, 31.3524, 38.5889, 44.1105, 49.8849, 55.5297, 64.6479, 72.7471, 84.6595, 90.4970, 99.9710, 111.6817, 120.8941, 234.9739, 391.9157, 648.5382, 920.2396, 1367.7080, 1647.1145, 2440.9581, 3028.6825, 3518.1483, 4601.6684, 5660.8254, 6723.2414, 7085.6261 ]
compute_ZSTD_l1_disk = [ 0.0022, 0.0300, 0.1066, 0.4099, 0.8974, 1.3218, 1.7679, 2.2154, 2.7007, 3.4267, 3.9255, 4.4597, 5.1155, 5.8251, 6.2064, 8.0141, 8.4316, 9.3195, 9.4570, 10.7034, 11.9192, 22.1895, 36.6542, 66.7209, 89.2111, 126.3853, 155.7241, 203.4894, 288.3248, 352.8067, 383.0908, 478.0074, 545.8722, 657.2160 ]

create_numpy = [ 0.0020, 0.0527, 0.2292, 0.9412, 2.1043, 2.8286, 3.7046, 4.7217, 5.8308, 7.0491 ]
compute_numpy = [ 0.0179, 0.2495, 0.9840, 3.9263, 8.8450, 12.0259, 16.3507, 40.1672, 155.1292, 302.5115 ]

create_numpy_jit = [ 0.0019, 0.0529, 0.2261, 0.9219, 2.0589, 2.8350, 3.7131, 18.4375, 26.5959, 34.5221, 33.7157, 49.6762, 63.1401 ]
compute_numpy_jit = [ 0.0035, 0.0180, 0.0622, 0.2307, 0.5196, 0.7095, 0.9251, 1.1981, 1.4729, 2.2007, 2.0953, 12.6746, 26.6424 ]


yaxis_title = 'Time (s)'
xaxis_type = 'log'
#xaxis_type = 'linear'
x64 = 64
alt_tit = ""
if xaxis_type == 'log':
    x64 = np.log10(64)
else:
    # We don't want to plot small values in the x-axis, so let's use th multiples of 50 in sizes
    alt_tit = "(**beyond RAM**)"
    sizes_ = []
    create_LZ4_l1_ = []
    compute_LZ4_l1_ = []
    create_LZ4_l1_disk_ = []
    compute_LZ4_l1_disk_ = []
    create_ZSTD_l1_ = []
    compute_ZSTD_l1_ = []
    create_ZSTD_l1_disk_ = []
    compute_ZSTD_l1_disk_ = []
    for size in sizes:
        if size % 50 == 0:
            # Find the position of the size in the original list
            pos = sizes.index(size)
            sizes_.append(size)
            create_LZ4_l1_.append(create_LZ4_l1[pos])
            compute_LZ4_l1_.append(compute_LZ4_l1[pos])
            create_LZ4_l1_disk_.append(create_LZ4_l1_disk[pos])
            compute_LZ4_l1_disk_.append(compute_LZ4_l1_disk[pos])
            create_ZSTD_l1_.append(create_ZSTD_l1[pos])
            compute_ZSTD_l1_.append(compute_ZSTD_l1[pos])
            create_ZSTD_l1_disk_.append(create_ZSTD_l1_disk[pos])
            compute_ZSTD_l1_disk_.append(compute_ZSTD_l1_disk[pos])
    sizes = np.array(sizes_)
    sizes_GB = np.array([n * 1000 * n * 1000 * 8 * 2 / 2 ** 30 for n in sizes])
    create_LZ4_l1 = create_LZ4_l1_
    compute_LZ4_l1 = compute_LZ4_l1_
    create_LZ4_l1_disk = create_LZ4_l1_disk_
    compute_LZ4_l1_disk = compute_LZ4_l1_disk_
    create_ZSTD_l1 = create_ZSTD_l1_
    compute_ZSTD_l1 = compute_ZSTD_l1_
    create_ZSTD_l1_disk = create_ZSTD_l1_disk_
    compute_ZSTD_l1_disk = compute_ZSTD_l1_disk_


if iobw:
    yaxis_title = 'I/O bandwidth (GB/s)'
    # Convert times to I/O bandwidth
    if xaxis_type == 'log':
        create_l0 = sizes_GB[:len(create_l0)] / np.array(create_l0)
        compute_l0 = sizes_GB[:len(compute_l0)] / np.array(compute_l0)
        create_l0_disk = sizes_GB[:len(create_l0_disk)] / np.array(create_l0_disk)
        compute_l0_disk = sizes_GB[:len(compute_l0_disk)] / np.array(compute_l0_disk)
        create_numpy = sizes_GB[:len(create_numpy)] / np.array(create_numpy)
        compute_numpy = sizes_GB[:len(compute_numpy)] / np.array(compute_numpy)
        create_numpy_jit = sizes_GB[:len(create_numpy_jit)] / np.array(create_numpy_jit)
        compute_numpy_jit = sizes_GB[:len(compute_numpy_jit)] / np.array(compute_numpy_jit)
    create_LZ4_l1 = sizes_GB[:len(create_LZ4_l1)] / np.array(create_LZ4_l1)
    compute_LZ4_l1 = sizes_GB[:len(compute_LZ4_l1)] / np.array(compute_LZ4_l1)
    create_LZ4_l1_disk = sizes_GB[:len(create_LZ4_l1_disk)] / np.array(create_LZ4_l1_disk)
    compute_LZ4_l1_disk = sizes_GB[:len(compute_LZ4_l1_disk)] / np.array(compute_LZ4_l1_disk)
    create_ZSTD_l1 = sizes_GB[:len(create_ZSTD_l1)] / np.array(create_ZSTD_l1)
    compute_ZSTD_l1 = sizes_GB[:len(compute_ZSTD_l1)] / np.array(compute_ZSTD_l1)
    create_ZSTD_l1_disk = sizes_GB[:len(create_ZSTD_l1_disk)] / np.array(create_ZSTD_l1_disk)
    compute_ZSTD_l1_disk = sizes_GB[:len(compute_ZSTD_l1_disk)] / np.array(compute_ZSTD_l1_disk)

def add_ram_limit(figure, compute=True):
    y1_max = 25 if compute else 2
    #y1_max = 35 if compute else y1_max
    figure.add_shape(
        type="line", x0=64, y0=0, x1=64, y1=y1_max,
        line=dict(color="Gray", width=2, dash="dot"),
    )
    figure.add_annotation(x=x64, y=y1_max * .9, text="64 GB RAM", showarrow=True, arrowhead=2, ax=45, ay=0, xref='x')

# Plot the data. There will be 2 plots: one for create times and another for compute times
labels = dict(
    l0="No compression", BLOSCLZ_l7="BLOSCLZ lvl=7", LZ4_l1="LZ4 lvl=1", ZSTD_l1="ZSTD lvl=1",
    numpy="NumPy", numpy_jit="NumPy (jit)"
)

# The create times plot
fig_create = go.Figure()
if xaxis_type == 'log':
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_l0, mode='lines+markers', name=labels["l0"] + " (mem)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_l0_disk, mode='lines+markers', name=labels["l0"] + " (disk)"))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_LZ4_l1, mode='lines+markers', name=labels["LZ4_l1"] + " (mem)"))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_LZ4_l1_disk, mode='lines+markers', name=labels["LZ4_l1"] + " (disk)"))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l1, mode='lines+markers', name=labels["ZSTD_l1"] + " (mem)"))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l1_disk, mode='lines+markers', name=labels["ZSTD_l1"] + " (disk)"))
if xaxis_type == 'log':
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_numpy_jit, mode='lines+markers',
                   name=labels["numpy"] + " (mem)", line=dict(color='brown')))
fig_create.update_layout(title=f'Create operands {alt_tit}: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title,
                         xaxis_type=xaxis_type)

# Add a vertical line at RAM limit
add_ram_limit(fig_create, compute=False)

# The compute times plot
# Calculate the maximum y1 value
y1_max = max(max(compute_l0), max(compute_l0_disk), max(compute_LZ4_l1), max(compute_LZ4_l1_disk),
             max(compute_ZSTD_l1), max(compute_ZSTD_l1_disk), max(compute_numpy), max(compute_numpy_jit))

fig_compute = go.Figure()
if xaxis_type == 'log':
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_l0, mode='lines+markers', name=labels["l0"] + " (mem)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_l0_disk, mode='lines+markers', name=labels["l0"] + " (disk)"))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_LZ4_l1, mode='lines+markers', name=labels["LZ4_l1"] + " (mem)"))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_LZ4_l1_disk, mode='lines+markers', name=labels["LZ4_l1"] + " (disk)"))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l1, mode='lines+markers', name=labels["ZSTD_l1"] + " (mem)"))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l1_disk, mode='lines+markers', name=labels["ZSTD_l1"] + " (disk)"))
if xaxis_type == 'log':
    fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy, mode='lines+markers',
                                     name=labels["numpy"], line=dict(color='brown')))
#fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy_jit, mode='lines+markers', name=labels["numpy_jit"]))
fig_compute.update_layout(title=f'Blosc2 compute {alt_tit}: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title,
                        xaxis_type=xaxis_type)

# Add a vertical line at RAM limit
add_ram_limit(fig_compute, compute=True)

# Show the plots
fig_create.show()
fig_compute.show()
