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

plotly = True

sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
sizes_GB = [n * 1000 * n * 1000 * 8 * 2 / 2**30 for n in sizes]

intel = False
amd = True
m2linux = False

# Load the data
if amd:
    title_ = "AMD Ryzen 9 9800X3D (64 GB RAM)"
    title_ = "np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)"

    create_clevel0 = [ 0.0325, 0.2709, 1.0339, 4.0489, 9.0849, 12.4154, 16.7818, 25.5946, 47.5691, 35.9919, 45.4295, 93.3075, 66.6529 ]
    compute_clevel0 = [ 0.0017, 0.0243, 0.0869, 0.3370, 0.7665, 1.0375, 1.3727, 1.7377, 2.1472, 2.6205, 3.0435, 18.5878, 28.0816 ]

    create_clevel0_ooc = [ 0.0305, 0.3371, 1.3249, 5.0602, 11.0410, 16.3685, 22.2012, 27.1348, 31.7409, 38.0690, 47.4424, 56.9335, 62.6965 ]
    compute_clevel0_ooc = [ 0.0019, 0.0243, 0.0885, 0.3434, 0.7761, 1.0724, 1.4082, 1.7373, 2.1827, 2.6124, 7.0940, 9.0734, 10.1089 ]

    create_LZ4_l1 = [ 0.0304, 0.2582, 1.0298, 3.9502, 8.8945, 11.9267, 16.3965, 20.2368, 24.6837, 29.3425, 36.2631, 42.1709, 48.0605, 52.3962, 61.5175, 68.6328, 80.1160, 85.4322, 97.1122, 106.9973, 114.8584 ]
    compute_LZ4_l1 = [ 0.0018, 0.0210, 0.0756, 0.3003, 0.6609, 0.8886, 1.1285, 1.4453, 1.7959, 2.1889, 2.6978, 3.1586, 3.4286, 3.9929, 4.4590, 5.3601, 5.6702, 6.4690, 6.9764, 7.8714, 8.6404 ]

    create_LZ4_l1_ooc = [ 1.7980, 0.2617, 1.0480, 4.0809, 9.0720, 13.8294, 16.7269, 20.5108, 24.9465, 30.0428, 37.1903, 42.8075, 48.7775, 52.9890, 63.4071, 70.1766, 81.9747, 88.1830, 97.7921, 111.0611, 119.7673 ]
    compute_LZ4_l1_ooc = [ 0.0019, 0.0214, 0.0795, 0.3060, 0.6985, 0.9195, 1.1766, 1.5213, 1.8845, 2.2972, 2.8044, 3.2587, 3.5898, 4.1524, 4.6293, 5.5485, 5.8715, 6.7386, 7.3019, 8.2307, 9.0145 ]

    create_ZSTD_l1 = [ 0.0302, 0.2704, 1.0703, 4.1243, 9.2185, 12.5026, 17.0585, 20.8708, 25.5844, 31.0571, 37.7114, 42.8297, 50.2696, 54.5773, 63.6311, 73.0370, 84.0092, 89.0686, 100.3300, 108.8173, 119.1154 ]
    compute_ZSTD_l1 = [ 0.0021, 0.0296, 0.1045, 0.3979, 0.8787, 1.3064, 1.7404, 2.1938, 2.6780, 3.3929, 3.8601, 4.3665, 5.0127, 5.7346, 6.1056, 7.9448, 8.2872, 9.4659, 9.2376, 10.4273, 11.6572 ]

    create_ZSTD_l1_ooc = [ 0.6564, 0.2825, 1.0826, 4.1968, 9.5022, 13.4840, 17.5387, 21.5807, 26.0052, 31.3524, 38.5889, 44.1105, 49.8849, 55.5297, 64.6479, 72.7471, 84.6595, 90.4970, 99.9710, 111.6817, 120.8941 ]
    compute_ZSTD_l1_ooc = [ 0.0022, 0.0300, 0.1066, 0.4099, 0.8974, 1.3218, 1.7679, 2.2154, 2.7007, 3.4267, 3.9255, 4.4597, 5.1155, 5.8251, 6.2064, 8.0141, 8.4316, 9.3195, 9.4570, 10.7034, 11.9192 ]

    create_numpy = [ 0.0020, 0.0527, 0.2292, 0.9412, 2.1043, 2.8286, 3.7046, 4.7217, 5.8308, 7.0491 ]
    compute_numpy = [ 0.0179, 0.2495, 0.9840, 3.9263, 8.8450, 12.0259, 16.3507, 40.1672, 155.1292, 302.5115 ]

    create_numpy_jit = [ 0.0019, 0.0529, 0.2261, 0.9219, 2.0589, 2.8350, 3.7131, 18.4375, 26.5959, 34.5221, 33.7157, 49.6762, 63.1401 ]
    compute_numpy_jit = [ 0.0035, 0.0180, 0.0622, 0.2307, 0.5196, 0.7095, 0.9251, 1.1981, 1.4729, 2.2007, 2.0953, 12.6746, 26.6424 ]

elif intel:
    title_ = "Intel Core i9-13900K (32 GB RAM)"
    create_clevel0 = [ 0.1810, 0.3511, 1.1511, 4.4575, 10.3164, 17.4344, 24.4274, 37.7116, 36.6179, 53.7264 ]
    compute_clevel0 = [ 0.0045, 0.0133, 0.0506, 0.2086, 0.4603, 0.8689, 1.1458, 1.4150, 1.7656, 1.9475 ]

    create_LZ4_l1 = [ 0.1834, 0.3457, 1.1234, 4.3301, 10.0406, 16.9509, 22.1617, 26.3818, 32.4472, 39.3830, 41.9484, 52.6316 ]
    compute_LZ4_l1 = [ 0.0014, 0.0128, 0.0494, 0.1958, 0.4387, 0.8207, 1.0208, 1.2739, 1.5062, 1.7446, 2.1553, 2.4458 ]

    create_ZSTD_l1 = [ 0.0362, 0.3734, 1.2009, 4.5362, 10.3706, 18.7104, 23.1148, 27.6572, 33.7207, 41.0326, 44.2322, 54.9467 ]
    compute_ZSTD_l1 = [ 0.0028, 0.0193, 0.0799, 0.2226, 0.4983, 0.9072, 1.1624, 1.4375, 1.8162, 2.0918, 2.5067, 2.7760 ]

    create_numpy = [ 0.0046, 0.1160, 0.4327, 1.7166, 3.8661, 7.5005, 14.1090, 18.6720, 64.2425, 108.4532, 529.1393, 962.1662 ]
    compute_numpy = [ 0.0240, 0.1920, 0.7217, 2.9316, 6.5893, 14.0353, 47.4275, 99.0893, 187.8040, 202.3973, 460.5915, 551.2776 ]

elif m2linux:
    title_ = "MacBook Air M2 (24 GB RAM)"
    create_LZ4_l1 = [0.021555185317993164, 0.2862977981567383, 0.8696625232696533, 3.4979920387268066, 8.235799789428711, 13.708781242370605, 20.74394702911377, 33.23137378692627]
    compute_LZ4_l1 = [0.0033464431762695312, 0.03627762794494629, 0.14009513854980468, 0.5438736915588379, 1.2493964672088622, 2.194223642349243, 3.5851136207580567, 5.067658472061157]

    create_numpy = [0.0016903877258300781, 0.04910874366760254, 0.18264532089233398, 0.7124006748199463, 1.8350563049316406, 8.877023935317993, 101.2457287311554, 196.21723294258118]
    compute_numpy = [0.003887462615966797, 0.026979732513427734, 0.11047358512878418, 0.4213367462158203, 0.9288184165954589, 1.6470709323883057, 5.5601390361785885, 9.401740503311157]

else:
    title_ = "Mac Mini M4 Pro (24 GB RAM)"

    create_numpy = [ 0.0024, 0.0686, 0.2857, 1.1800, 3.5006, 13.7092, 21.9491, 30.5237, 101.3553, 363.5005, 446.5876, 1509.1826 ]
    compute_numpy = [ 0.0046, 0.1173, 0.5066, 2.0908, 5.6268, 13.0679, 16.7926, 20.9192, 25.5899, 34.5382, 46.0664, 1083.7046 ]

    create_numpy_jit = [ 0.0024, 0.0686, 0.2857, 1.1800, 3.5006, 13.7092, 21.9491, 30.5237, 101.3553, 363.5005, 446.5876, 1509.1826 ]
    compute_numpy_jit = [  ]

    create_clevel0 = [ 0.0321, 0.6742, 1.6750, 5.1412, 12.4538, 20.6695, 28.0185, 34.7422, 41.3935, 50.4275, 59.5572, 71.3740 ]
    compute_clevel0 = [ 0.0017, 0.0224, 0.0761, 0.2439, 0.5827, 0.9816, 1.2265, 6.8712, 9.6391, 11.2875, 13.1953, 15.4047 ]

    create_LZ4_l1 = [ 0.0316, 0.6974, 1.6957, 5.2077, 11.8975, 20.2346, 26.6419, 32.7686, 38.2088, 47.5221, 55.7224, 66.6138 ]
    compute_LZ4_l1 = [ 0.0019, 0.0234, 0.0763, 0.2421, 0.5533, 0.9384, 1.2476, 1.5299, 1.8564, 2.1836, 2.5633, 2.9245 ]

    create_ZSTD_l1 = [ 0.0354, 0.7105, 1.7217, 5.2663, 12.7385, 20.3693, 28.1353, 33.0310, 40.3843, 50.2020, 58.0643, 69.5190 ]
    compute_ZSTD_l1 = [ 0.0021, 0.0225, 0.0773, 0.2630, 0.8735, 1.0087, 1.9553, 1.5670, 3.1986, 3.3499, 4.0728, 4.6287 ]


# Plot the data. There will be 2 plots: one for create times and another for compute times
labels = ['lvl=0', 'LZ4 lvl=1', 'ZSTD lvl=1', 'NumPy', "numpy_jit"]
if plotly:
    # Create the create times plot
    fig_create = go.Figure()
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_clevel0, mode='lines+markers', name=labels[0]))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_clevel0_ooc, mode='lines+markers', name=labels[0] + "(ooc)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_LZ4_l1, mode='lines+markers', name=labels[1]))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_LZ4_l1_ooc, mode='lines+markers', name=labels[1] + "(ooc)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_ZSTD_l1, mode='lines+markers', name=labels[2]))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_ZSTD_l1_ooc, mode='lines+markers', name=labels[2] + "(ooc)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_numpy_jit, mode='lines+markers', name=labels[3]))
    fig_create.update_layout(title=f'Create operands times ({title_})', xaxis_title='Size (GB)', yaxis_title='Time (s)')

    # Create the compute times plot
    # Calculate the maximum y1 value
    y1_max = max(max(compute_clevel0), max(compute_clevel0_ooc), max(compute_LZ4_l1), max(compute_LZ4_l1_ooc),
                 max(compute_ZSTD_l1), max(compute_ZSTD_l1_ooc), max(compute_numpy), max(compute_numpy_jit))

    fig_compute = go.Figure()
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_clevel0, mode='lines+markers', name=labels[0]))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_clevel0_ooc, mode='lines+markers', name=labels[0] + "(ooc)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_LZ4_l1, mode='lines+markers', name=labels[1]))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_LZ4_l1_ooc, mode='lines+markers', name=labels[1] + "(ooc)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_ZSTD_l1, mode='lines+markers', name=labels[2]))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_ZSTD_l1_ooc, mode='lines+markers', name=labels[2] + "(ooc)"))
    fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy, mode='lines+markers', name=labels[3]))
    fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy_jit, mode='lines+markers', name=labels[4]))
    fig_compute.update_layout(title=f'Compute times ({title_})', xaxis_title='Size (GB)', yaxis_title='Time (s)')

    # Add a vertical line at 64 GB
    y1_max = 35
    fig_compute.add_shape(
        type="line", x0=64, y0=0, x1=64, y1=y1_max,
        line=dict(color="Gray", width=2, dash="dot"),
    )
    if amd:
        fig_compute.add_annotation(x=64, y=y1_max * .9, text="64 GB", showarrow=True, arrowhead=2, ax=40, ay=0)

    # Show the plots
    fig_create.show()
    fig_compute.show()
else:
    plt.figure()
    plt.plot(sizes_GB, create_clevel0, "o-", label=labels[0])
    plt.plot(sizes_GB, create_LZ4_l1, "o-", label=labels[1])
    plt.plot(sizes_GB, create_ZSTD_l1, "o-", label=labels[2])
    plt.plot(sizes_GB, create_numpy_jit, "o-", label=labels[3])
    plt.xlabel("Size (GB)")
    plt.ylabel("Time (s)")
    plt.title(f"Create times ({title_})")
    plt.legend()
    # Now, the compute times
    plt.figure()
    plt.plot(sizes_GB, compute_clevel0, "o-", label=labels[0])
    plt.plot(sizes_GB, compute_LZ4_l1, "o-", label=labels[1])
    plt.plot(sizes_GB, compute_ZSTD_l1, "o-", label=labels[2])
    plt.plot(sizes_GB, compute_numpy, "o-", label=labels[3])
    plt.plot(sizes_GB, compute_numpy_jit, "o-", label=labels[4])
    plt.xlabel("Size (GB)")
    plt.ylabel("Time (s)")
    plt.title(f"Compute times ({title_})")
    plt.legend()
    plt.show()
