#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Plots for the jit vs. numpy benchmarks on different array sizes and platforms.

import plotly.graph_objects as go
import numpy as np

iobw = True  # use I/O bandwidth instead of time

sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]
sizes_GB = np.array([n * 1000 * n * 1000 * 4 * 2 / 2**30 for n in sizes])

# Default title
title_ = "np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1); (codec: ZSTD)"

# Load the data from AMD Ryzen 9 9800X3D (64 GB RAM)
#title_ = "AMD Ryzen 9 9800X3D (64 GB RAM)"

create_ZSTD_l5_8bits_disk = [ 0.0291, 0.3015, 1.0396, 4.3120, 9.4448, 11.9615, 16.4934, 20.8363, 25.6686, 30.5084, 37.4541, 43.1708, 49.5912, 54.8510, 62.9904, 71.6792, 82.8624, 87.3148, 99.6089, 110.6020, 120.6817, 230.7189, 393.3838, 635.6783, 920.4081, 1224.8611, 1542.1973, 2067.7355, 2643.4060, 3960.7069, 6605.8679 ]
compute_ZSTD_l5_8bits_disk = [ 0.0018, 0.0264, 0.0666, 0.3514, 0.5839, 0.7897, 1.0354, 1.3110, 1.6365, 1.9557, 2.3461, 2.7590, 3.1654, 3.6511, 4.1705, 4.6487, 5.2456, 5.9307, 6.6057, 7.1372, 7.8886, 14.4919, 26.9140, 41.5376, 59.6396, 79.8878, 109.3518, 134.7697, 167.8493, 242.3677, 328.7269 ]

create_ZSTD_l5_8bits_mem = [ 0.2848, 0.5540, 1.6162, 5.0427, 10.2004, 14.9469, 17.0872, 23.2580, 26.4399, 35.9111, 38.8774, 47.2819, 59.8694, 55.6182, 64.7790, 73.3225, 89.1435, 89.1889, 105.3143, 123.7543, 127.4739, 268.2381, 397.1528, 682.4370, 931.2079, 1408.0286, 1907.0228, 2513.9356, 3169.7178, 4898.9904, 6108.3949 ]
compute_ZSTD_l5_8bits_mem = [ 3.5426, 0.0439, 0.0721, 0.3544, 0.6075, 0.7633, 1.0329, 1.2853, 1.6016, 1.9229, 2.2995, 2.7300, 3.1072, 3.5914, 4.0754, 4.5324, 5.1152, 5.8040, 6.4044, 6.9661, 7.7495, 14.3803, 26.1613, 40.5647, 58.5311, 77.8399, 105.7455, 132.6907, 166.3500, 247.3172, 325.7362 ]

create_ZSTD_l5_12bits_disk = [ 0.0431, 0.2961, 1.0377, 4.3224, 9.1700, 11.9641, 16.5006, 20.8539, 25.5999, 30.5143, 37.1139, 43.6415, 49.8283, 54.4649, 63.6562, 71.0058, 82.8709, 87.4242, 99.8155, 110.6995, 120.7145, 228.1858, 388.0447, 630.0056, 901.3052, 1227.0249, 1538.4994, 2192.4736, 3058.9535, 3970.1224, 6720.8534 ]
compute_ZSTD_l5_12bits_disk = [ 0.0018, 0.0261, 0.0668, 0.3529, 0.5862, 0.8014, 1.0288, 1.3392, 1.6499, 1.9708, 2.3465, 2.8174, 3.1577, 3.6683, 4.2046, 4.6664, 5.2713, 5.9672, 6.6033, 7.3388, 7.9277, 14.7204, 26.9279, 41.8064, 59.8765, 80.5294, 108.8107, 136.0069, 169.9042, 242.5698, 334.2899 ]

create_ZSTD_l5_12bits_mem = [ 0.3097, 0.7280, 1.5824, 5.3017, 10.0199, 15.0386, 16.8093, 23.5793, 26.4025, 35.4388, 38.3893, 47.5386, 61.0661, 56.6073, 66.3175, 72.9117, 89.0572, 89.0964, 104.8680, 125.0135, 128.6147, 269.4906, 397.8105, 743.2941, 936.2004, 1440.9327, 1934.9108, 2547.0800, 3438.9840, 4912.5360, 6103.0010 ]
compute_ZSTD_l5_12bits_mem = [ 3.2933, 0.0450, 0.0823, 0.3598, 0.6156, 0.7802, 1.0374, 1.3120, 1.6274, 1.9737, 2.3018, 2.7496, 3.0923, 3.6573, 4.0879, 4.5646, 5.1826, 5.8380, 6.4389, 7.0517, 7.7586, 14.6210, 26.0380, 40.9289, 58.5579, 79.5502, 106.3976, 134.6044, 166.4742, 237.0625, 333.7096 ]

create_ZSTD_l5_16bits_disk = [ 0.0430, 0.3144, 1.0715, 4.2417, 9.1328, 11.9006, 16.4920, 20.4754, 25.5973, 30.1237, 36.9232, 42.6159, 48.9959, 53.9110, 62.4312, 70.7186, 81.0649, 86.1593, 98.1041, 110.4069, 120.2413, 226.6709, 381.0409, 620.3338, 892.2901, 1240.1823, 1629.5867, 2177.8013, 2969.3828, 3967.6243, 6609.0145 ]
compute_ZSTD_l5_16bits_disk = [ 0.0018, 0.0271, 0.0691, 0.3559, 0.5969, 0.8219, 1.0591, 1.3476, 1.6760, 1.9941, 2.3686, 2.8510, 3.1904, 3.7279, 4.2099, 4.7084, 5.3074, 5.9957, 6.6762, 7.2743, 8.0519, 14.8181, 27.5201, 42.3674, 60.4739, 81.2832, 112.6656, 139.3029, 174.4497, 246.2020, 336.9309 ]

create_ZSTD_l5_16bits_mem = [ 0.2618, 0.9147, 1.6346, 5.2474, 10.0476, 15.1650, 17.5610, 22.8673, 26.4274, 36.0352, 39.2973, 47.8204, 60.1208, 55.8942, 68.1996, 73.0547, 85.7855, 89.3090, 104.9364, 126.9699, 123.1824, 276.0629, 396.0899, 743.5490, 934.4396, 1478.9950, 1931.2574, 2532.6307, 3402.5700, 4885.8968, 6654.6702 ]
compute_ZSTD_l5_16bits_mem = [ 2.7690, 0.0459, 0.0738, 0.3657, 0.6195, 0.7958, 1.0575, 1.3256, 1.6513, 2.0090, 2.3522, 2.8353, 3.1545, 3.6713, 4.1154, 4.6581, 5.2726, 5.9554, 6.5680, 7.2284, 7.9461, 14.6099, 26.8851, 41.2897, 59.4072, 79.7960, 108.8751, 136.0794, 169.3939, 240.8721, 338.8543 ]

create_ZSTD_l5_24bits_disk = [ 0.0443, 0.3082, 1.1479, 4.6196, 10.3190, 13.5080, 18.2468, 22.4225, 28.2498, 33.4455, 40.8569, 46.7288, 53.3009, 59.0729, 67.3034, 75.1796, 87.8337, 92.0496, 103.9884, 115.9744, 127.8724, 234.0250, 399.6466, 643.1328, 922.7186, 1243.4742, 1585.6460, 2392.9311, 3028.6756, 4026.1285, 6778.0339 ]
compute_ZSTD_l5_24bits_disk = [ 0.0018, 0.0275, 0.0743, 0.3770, 0.6462, 0.8711, 1.1622, 1.4456, 1.8297, 2.1455, 2.5582, 3.0233, 3.4629, 3.9855, 4.5493, 5.0444, 5.7148, 6.4459, 7.0735, 7.6082, 8.5538, 15.6118, 28.4247, 43.4489, 62.4833, 84.2844, 112.2303, 145.1725, 175.4419, 250.6996, 342.5847 ]

create_ZSTD_l5_24bits_mem = [ 0.2846, 0.7443, 1.7465, 5.6776, 11.1323, 16.4522, 19.0117, 25.3204, 28.7993, 38.0313, 41.5742, 50.2025, 63.5148, 60.0686, 70.0280, 76.9878, 95.6996, 93.7957, 108.3032, 130.7858, 131.2840, 274.0678, 405.2104, 748.1952, 955.4778, 1448.5087, 1947.1579, 2444.6069, 3487.4620, 4914.8358, 6685.2610 ]
compute_ZSTD_l5_24bits_mem = [ 2.7509, 0.0466, 0.0854, 0.3774, 0.6809, 0.8508, 1.1446, 1.4176, 1.8078, 2.1420, 2.5089, 2.9998, 3.4242, 3.9550, 4.4779, 4.9172, 5.5910, 6.2880, 6.9373, 7.5398, 8.3317, 15.4684, 27.6829, 43.0323, 61.5013, 82.3317, 110.7303, 140.3281, 173.7906, 248.9565, 340.5203 ]

create_ZSTD_l5_32bits_disk = [ 0.0515, 0.3116, 1.1512, 4.6352, 9.9101, 12.8060, 17.4190, 22.0518, 27.1258, 32.9336, 40.1834, 45.2333, 51.1433, 57.7009, 66.7316, 75.2147, 86.7955, 92.3465, 112.3666, 123.4091, 136.4982, 248.1517, 425.7151, 692.5093, 964.6273, 1288.3537, 1768.9565, 2363.8556, 3052.0195, 4435.4477, 7077.3454 ]
compute_ZSTD_l5_32bits_disk = [ 0.0020, 0.0297, 0.0831, 0.4003, 0.7244, 0.9587, 1.2656, 1.5964, 1.9488, 2.3584, 2.8733, 3.3516, 3.6925, 4.4041, 4.9282, 5.7000, 6.0026, 7.1855, 7.6281, 8.3859, 9.1505, 16.9414, 31.5513, 47.6357, 66.7197, 88.0063, 124.9729, 157.2028, 189.8281, 277.6650, 372.2259 ]

create_ZSTD_l5_32bits_mem = [ 0.2740, 1.0527, 1.8111, 5.7117, 11.1256, 16.2087, 18.1610, 25.3161, 27.6771, 38.2490, 40.6003, 48.3606, 63.4355, 59.2211, 68.5838, 76.7220, 97.3595, 94.7692, 117.2560, 138.9500, 139.6692, 292.7232, 430.8667, 796.4217, 1008.3740, 1488.9369, 2143.2650, 2772.4041, 3675.6133, 5406.4743, 6472.9431 ]
compute_ZSTD_l5_32bits_mem = [ 3.7490, 0.0531, 0.0929, 0.4056, 0.7479, 0.9526, 1.2541, 1.5919, 1.9174, 2.3495, 2.8487, 3.3340, 3.6438, 4.3989, 4.8966, 5.6537, 5.9449, 7.0888, 7.5406, 8.3222, 9.1164, 16.8181, 30.9136, 47.1204, 66.4587, 87.1323, 122.4179, 157.8325, 187.5637, 271.5958, 374.7192 ]

create_ZSTD_l5_f32_disk = [ 0.1891, 0.2530, 0.9717, 3.9297, 8.7861, 11.7525, 16.2177, 19.6682, 22.5171, 27.8397, 34.4995, 41.3844, 47.3245, 50.1879, 61.5008, 63.4198, 77.2572, 107.3055, 95.6815, 103.9656, 110.2893, 195.1330, 378.3826, 533.2835, 873.7248, 1151.9387, 1498.3907, 1954.9378, 2343.6427, 3477.0688, 4274.8765 ]
compute_ZSTD_l5_f32_disk = [ 0.0013, 0.0150, 0.0526, 0.2082, 0.4613, 0.6286, 0.8218, 1.0329, 1.2733, 1.5246, 1.7876, 2.1808, 2.4450, 2.7508, 3.1495, 3.5895, 3.9414, 4.4979, 4.9185, 5.4491, 5.9133, 10.9502, 19.4659, 30.3280, 43.5058, 59.6969, 78.8010, 98.6456, 123.3424, 174.8172, 238.0731 ]


yaxis_title = 'Time (s)'
if iobw:
    yaxis_title = 'I/O bandwidth (GB/s)'
    # Convert times to I/O bandwidth
    create_ZSTD_l5_8bits_disk = sizes_GB[:len(create_ZSTD_l5_8bits_disk)] / np.array(create_ZSTD_l5_8bits_disk)
    compute_ZSTD_l5_8bits_disk = sizes_GB[:len(compute_ZSTD_l5_8bits_disk)] / np.array(compute_ZSTD_l5_8bits_disk)
    create_ZSTD_l5_8bits_mem = sizes_GB[:len(create_ZSTD_l5_8bits_mem)] / np.array(create_ZSTD_l5_8bits_mem)
    compute_ZSTD_l5_8bits_mem = sizes_GB[:len(compute_ZSTD_l5_8bits_mem)] / np.array(compute_ZSTD_l5_8bits_mem)
    create_ZSTD_l5_12bits_disk = sizes_GB[:len(create_ZSTD_l5_12bits_disk)] / np.array(create_ZSTD_l5_12bits_disk)
    compute_ZSTD_l5_12bits_disk = sizes_GB[:len(compute_ZSTD_l5_12bits_disk)] / np.array(compute_ZSTD_l5_12bits_disk)
    create_ZSTD_l5_12bits_mem = sizes_GB[:len(create_ZSTD_l5_12bits_mem)] / np.array(create_ZSTD_l5_12bits_mem)
    compute_ZSTD_l5_12bits_mem = sizes_GB[:len(compute_ZSTD_l5_12bits_mem)] / np.array(compute_ZSTD_l5_12bits_mem)
    create_ZSTD_l5_16bits_disk = sizes_GB[:len(create_ZSTD_l5_16bits_disk)] / np.array(create_ZSTD_l5_16bits_disk)
    compute_ZSTD_l5_16bits_disk = sizes_GB[:len(compute_ZSTD_l5_16bits_disk)] / np.array(compute_ZSTD_l5_16bits_disk)
    create_ZSTD_l5_16bits_mem = sizes_GB[:len(create_ZSTD_l5_16bits_mem)] / np.array(create_ZSTD_l5_16bits_mem)
    compute_ZSTD_l5_16bits_mem = sizes_GB[:len(compute_ZSTD_l5_16bits_mem)] / np.array(compute_ZSTD_l5_16bits_mem)
    create_ZSTD_l5_24bits_disk = sizes_GB[:len(create_ZSTD_l5_24bits_disk)] / np.array(create_ZSTD_l5_24bits_disk)
    compute_ZSTD_l5_24bits_disk = sizes_GB[:len(compute_ZSTD_l5_24bits_disk)] / np.array(compute_ZSTD_l5_24bits_disk)
    create_ZSTD_l5_24bits_mem = sizes_GB[:len(create_ZSTD_l5_24bits_mem)] / np.array(create_ZSTD_l5_24bits_mem)
    compute_ZSTD_l5_24bits_mem = sizes_GB[:len(compute_ZSTD_l5_24bits_mem)] / np.array(compute_ZSTD_l5_24bits_mem)
    create_ZSTD_l5_32bits_disk = sizes_GB[:len(create_ZSTD_l5_32bits_disk)] / np.array(create_ZSTD_l5_32bits_disk)
    compute_ZSTD_l5_32bits_disk = sizes_GB[:len(compute_ZSTD_l5_32bits_disk)] / np.array(compute_ZSTD_l5_32bits_disk)
    create_ZSTD_l5_32bits_mem = sizes_GB[:len(create_ZSTD_l5_32bits_mem)] / np.array(create_ZSTD_l5_32bits_mem)
    compute_ZSTD_l5_32bits_mem = sizes_GB[:len(compute_ZSTD_l5_32bits_mem)] / np.array(compute_ZSTD_l5_32bits_mem)
    create_ZSTD_l5_f32_disk = sizes_GB[:len(create_ZSTD_l5_f32_disk)] / np.array(create_ZSTD_l5_f32_disk)
    compute_ZSTD_l5_f32_disk = sizes_GB[:len(compute_ZSTD_l5_f32_disk)] / np.array(compute_ZSTD_l5_f32_disk)


def add_ram_limit(figure, compute=True):
    y1_max = 20 if compute else 1
    #y1_max = 35 if compute else y1_max
    figure.add_shape(
        type="line", x0=64, y0=0, x1=64, y1=y1_max,
        line=dict(color="Gray", width=2, dash="dot"),
    )
    figure.add_annotation(x=np.log10(64), y=y1_max * .9, text="64 GB", showarrow=True, arrowhead=2, ax=40, ay=0, xref='x')


# Plot the data. There will be 2 plots: one for create times and another for compute times
labels = {
    '8bits_disk': "8 bits, disk",
    '8bits_mem': "8 bits, mem",
    '12bits_disk': "12 bits, disk",
    '12bits_mem': "12 bits, mem",
    '16bits_disk': "16 bits, disk",
    '16bits_mem': "16 bits, mem",
    '24bits_disk': "24 bits, disk",
    '24bits_mem': "24 bits, mem",
    '32bits_disk': "32 bits, disk",
    '32bits_mem': "32 bits, mem",
    'f32_disk': "f32, disk",
    'f32_mem': "f32, mem",
}

# The create times plot
fig_create = go.Figure()
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_8bits_disk, mode='lines+markers', name=labels["8bits_disk"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_8bits_mem, mode='lines+markers', name=labels["8bits_mem"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_12bits_disk, mode='lines+markers', name=labels["12bits_disk"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_12bits_mem, mode='lines+markers', name=labels["12bits_mem"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_16bits_disk, mode='lines+markers', name=labels["16bits_disk"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_16bits_mem, mode='lines+markers', name=labels["16bits_mem"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_24bits_disk, mode='lines+markers', name=labels["24bits_disk"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_24bits_mem, mode='lines+markers', name=labels["24bits_mem"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_32bits_disk, mode='lines+markers', name=labels["32bits_disk"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_32bits_mem, mode='lines+markers', name=labels["32bits_mem"]))
fig_create.add_trace(
    go.Scatter(x=sizes_GB, y=create_ZSTD_l5_f32_disk, mode='lines+markers', name=labels["f32_disk"],
               line=dict(color='brown')))
#fig_create.add_trace(go.Scatter(x=sizes_GB, y=create_ZSTD_l5_f32_mem, mode='lines+markers', name=labels["f32_mem"]))
fig_create.update_layout(title=f'Create operands: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title,
                         xaxis_type="log")

# Add a vertical line at RAM limit
add_ram_limit(fig_create, compute=False)

# The compute times plot
fig_compute = go.Figure()
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_8bits_disk, mode='lines+markers', name=labels["8bits_disk"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_8bits_mem, mode='lines+markers', name=labels["8bits_mem"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_12bits_disk, mode='lines+markers', name=labels["12bits_disk"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_12bits_mem, mode='lines+markers', name=labels["12bits_mem"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_16bits_disk, mode='lines+markers', name=labels["16bits_disk"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_16bits_mem, mode='lines+markers', name=labels["16bits_mem"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_24bits_disk, mode='lines+markers', name=labels["24bits_disk"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_24bits_mem, mode='lines+markers', name=labels["24bits_mem"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_32bits_disk, mode='lines+markers', name=labels["32bits_disk"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_32bits_mem, mode='lines+markers', name=labels["32bits_mem"]))
fig_compute.add_trace(
    go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_f32_disk, mode='lines+markers', name=labels["f32_disk"],
               line=dict(color='brown')))
#fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_ZSTD_l5_f32_mem, mode='lines+markers', name=labels["f32_mem"]))
fig_compute.update_layout(title=f'Blosc2 compute: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title,
                        xaxis_type="log")

# Add a vertical line at RAM limit
add_ram_limit(fig_compute, compute=True)

# Show the plots
fig_create.show()
fig_compute.show()
