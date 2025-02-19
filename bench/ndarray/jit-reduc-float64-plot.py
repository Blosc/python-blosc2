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

plotly = True
iobw = True  # use I/O bandwidth instead of time

sizes = [1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
sizes_GB = np.array([n * 1000 * n * 1000 * 8 * 2 / 2**30 for n in sizes])

amd = True
intel = False
m2linux = False

# Default title
title_ = "np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)"

# Load the data
if amd:
    #title_ = "AMD Ryzen 9 9800X3D (64 GB RAM)"

    create_l0 = [ 0.0325, 0.2709, 1.0339, 4.0489, 9.0849, 12.4154, 16.7818, 25.5946, 47.5691, 35.9919, 45.4295, 93.3075, 66.6529 ]
    compute_l0 = [ 0.0017, 0.0243, 0.0869, 0.3370, 0.7665, 1.0375, 1.3727, 1.7377, 2.1472, 2.6205, 3.0435, 18.5878, 28.0816 ]

    create_l0_disk = [ 0.0305, 0.3371, 1.3249, 5.0602, 11.0410, 16.3685, 22.2012, 27.1348, 31.7409, 38.0690, 47.4424, 56.9335, 62.6965, 65.2226, 81.1631, 92.8310, 103.7345, 112.1973, 124.5319 ]
    compute_l0_disk = [ 0.0019, 0.0243, 0.0885, 0.3434, 0.7761, 1.0724, 1.4082, 1.7373, 2.1827, 2.6124, 7.0940, 9.0734, 10.1089, 11.2911, 13.0464, 22.6369, 25.4538, 28.7107, 31.9562 ]

    create_BLOSCLZ_l7 = [ 0.0267, 0.2610, 1.0299, 3.9724, 9.1326, 11.7598, 16.0252, 20.1420, 24.7293, 33.8753, 37.2400, 41.9200, 48.4979, 53.1935, 61.3910, 70.3354, 79.8628, 84.3074, 95.8080, 107.0405, 117.4525 ]
    compute_BLOSCLZ_l7 = [ 0.0018, 0.0205, 0.0773, 0.2931, 0.6938, 0.9001, 1.1693, 1.4701, 1.8559, 3.3739, 2.7486, 3.2836, 3.5230, 4.1417, 4.8597, 5.5748, 5.9453, 6.9264, 7.3589, 8.3207, 9.1710 ]

    create_BLOSCLZ_l7_disk = [ 0.0701, 0.2656, 1.0553, 4.0486, 9.2255, 12.2674, 16.4618, 20.1527, 25.3657, 33.7537, 37.3551, 43.0586, 48.4968, 53.9183, 62.9415, 71.7656, 80.5597, 85.5704, 97.0770, 109.7463, 119.2675 ]
    compute_BLOSCLZ_l7_disk = [ 0.0019, 0.0213, 0.0788, 0.3002, 0.7252, 0.9276, 1.2053, 1.4999, 1.9109, 3.4081, 2.8205, 3.3593, 3.6086, 4.2295, 4.9548, 5.6996, 6.0085, 7.0802, 7.4786, 8.4466, 9.4861 ]

    create_LZ4_l1 = [ 0.0304, 0.2582, 1.0298, 3.9502, 8.8945, 11.9267, 16.3965, 20.2368, 24.6837, 29.3425, 36.2631, 42.1709, 48.0605, 52.3962, 61.5175, 68.6328, 80.1160, 85.4322, 97.1122, 106.9973, 114.8584 ]
    compute_LZ4_l1 = [ 0.0018, 0.0210, 0.0756, 0.3003, 0.6609, 0.8886, 1.1285, 1.4453, 1.7959, 2.1889, 2.6978, 3.1586, 3.4286, 3.9929, 4.4590, 5.3601, 5.6702, 6.4690, 6.9764, 7.8714, 8.6404 ]

    create_LZ4_l1_disk = [ 1.7980, 0.2617, 1.0480, 4.0809, 9.0720, 13.8294, 16.7269, 20.5108, 24.9465, 30.0428, 37.1903, 42.8075, 48.7775, 52.9890, 63.4071, 70.1766, 81.9747, 88.1830, 97.7921, 111.0611, 119.7673 ]
    compute_LZ4_l1_disk = [ 0.0019, 0.0214, 0.0795, 0.3060, 0.6985, 0.9195, 1.1766, 1.5213, 1.8845, 2.2972, 2.8044, 3.2587, 3.5898, 4.1524, 4.6293, 5.5485, 5.8715, 6.7386, 7.3019, 8.2307, 9.0145 ]

    create_ZSTD_l1 = [ 0.0302, 0.2704, 1.0703, 4.1243, 9.2185, 12.5026, 17.0585, 20.8708, 25.5844, 31.0571, 37.7114, 42.8297, 50.2696, 54.5773, 63.6311, 73.0370, 84.0092, 89.0686, 100.3300, 108.8173, 119.1154 ]
    compute_ZSTD_l1 = [ 0.0021, 0.0296, 0.1045, 0.3979, 0.8787, 1.3064, 1.7404, 2.1938, 2.6780, 3.3929, 3.8601, 4.3665, 5.0127, 5.7346, 6.1056, 7.9448, 8.2872, 9.4659, 9.2376, 10.4273, 11.6572 ]

    create_ZSTD_l1_disk = [ 0.6564, 0.2825, 1.0826, 4.1968, 9.5022, 13.4840, 17.5387, 21.5807, 26.0052, 31.3524, 38.5889, 44.1105, 49.8849, 55.5297, 64.6479, 72.7471, 84.6595, 90.4970, 99.9710, 111.6817, 120.8941 ]
    compute_ZSTD_l1_disk = [ 0.0022, 0.0300, 0.1066, 0.4099, 0.8974, 1.3218, 1.7679, 2.2154, 2.7007, 3.4267, 3.9255, 4.4597, 5.1155, 5.8251, 6.2064, 8.0141, 8.4316, 9.3195, 9.4570, 10.7034, 11.9192 ]

    create_numpy = [ 0.0020, 0.0527, 0.2292, 0.9412, 2.1043, 2.8286, 3.7046, 4.7217, 5.8308, 7.0491 ]
    compute_numpy = [ 0.0179, 0.2495, 0.9840, 3.9263, 8.8450, 12.0259, 16.3507, 40.1672, 155.1292, 302.5115 ]

    create_numpy_jit = [ 0.0019, 0.0529, 0.2261, 0.9219, 2.0589, 2.8350, 3.7131, 18.4375, 26.5959, 34.5221, 33.7157, 49.6762, 63.1401 ]
    compute_numpy_jit = [ 0.0035, 0.0180, 0.0622, 0.2307, 0.5196, 0.7095, 0.9251, 1.1981, 1.4729, 2.2007, 2.0953, 12.6746, 26.6424 ]

elif intel:
    title_ = "Intel Core i9-13900K (32 GB RAM)"
    create_l0 = [ 0.1810, 0.3511, 1.1511, 4.4575, 10.3164, 17.4344, 24.4274, 37.7116, 36.6179, 53.7264 ]
    compute_l0 = [ 0.0045, 0.0133, 0.0506, 0.2086, 0.4603, 0.8689, 1.1458, 1.4150, 1.7656, 1.9475 ]

    create_l0_disk = [0] * 10  # this crashed
    compute_l0_disk = [0] * 10  # this crashed

    create_LZ4_l1 = [ 0.1834, 0.3457, 1.1234, 4.3301, 10.0406, 16.9509, 22.1617, 26.3818, 32.4472, 39.3830, 41.9484, 52.6316 ]
    compute_LZ4_l1 = [ 0.0014, 0.0128, 0.0494, 0.1958, 0.4387, 0.8207, 1.0208, 1.2739, 1.5062, 1.7446, 2.1553, 2.4458 ]

    create_LZ4_l1_disk = [ 0.1222, 0.3705, 1.4912, 5.4410, 12.3593, 15.6122, 21.9754, 27.6554, 34.0044, 41.8007, 49.8841, 58.0062, 58.1169, 76.9802, 79.2385, 99.9344, 111.9739, 126.4542, 142.3726 ]
    compute_LZ4_l1_disk = [ 0.0032, 0.0167, 0.1319, 0.3058, 0.7025, 0.9334, 1.2293, 1.5071, 1.8350, 2.4390, 2.8756, 3.3668, 3.8927, 4.5542, 5.0557, 6.2732, 6.5550, 7.4660, 8.0298 ]

    create_ZSTD_l1 = [ 0.0362, 0.3734, 1.2009, 4.5362, 10.3706, 18.7104, 23.1148, 27.6572, 33.7207, 41.0326, 44.2322, 54.9467 ]
    compute_ZSTD_l1 = [ 0.0028, 0.0193, 0.0799, 0.2226, 0.4983, 0.9072, 1.1624, 1.4375, 1.8162, 2.0918, 2.5067, 2.7760 ]

    create_ZSTD_l1_disk = [ 0.0547, 0.4150, 1.5916, 5.7187, 13.3500, 16.8552, 23.2673, 29.2232, 35.5580, 44.3726, 52.4742, 59.8893, 61.3350, 80.1619, 83.0139, 103.8481, 117.3893, 128.6241, 138.2671 ]
    compute_ZSTD_l1_disk = [ 0.0031, 0.0213, 0.1465, 0.3784, 0.8567, 1.2848, 1.7557, 1.9248, 2.3045, 3.3080, 3.6730, 4.2439, 5.4268, 6.5462, 6.6983, 8.2491, 8.9797, 9.8748, 9.9348 ]

    create_numpy = [ 0.0035, 0.0784, 0.3107, 1.2150, 2.7350, 3.7511 ]
    compute_numpy = [ 0.0327, 0.3483, 1.3650, 5.4224, 14.3476, 80.2920 ]

    create_numpy_jit = [ 0.0035, 0.0785, 0.3088, 1.2377, 2.8435, 6.7555, 11.3731 ]
    compute_numpy_jit = [ 0.0043, 0.0164, 0.0564, 0.2203, 0.4830, 0.6645, 0.8571 ]

elif m2linux:
    title_ = "MacBook Air M2 (24 GB RAM)"

    create_l0 = [ 0.0444, 0.7885, 2.3555, 8.4279, 18.9511, 27.8466, 38.0111, 48.6637 ]
    compute_l0 = [ 0.0030, 0.0503, 0.1845, 0.7183, 1.5504, 8.5181, 11.1162, 48.3423 ]

    create_l0_disk = [ 0.1204, 0.8043, 2.6619, 8.9401, 21.9047, 29.0938, 36.9753, 45.9740 ]
    compute_l0_disk = [ 0.0038, 0.0733, 0.2713, 4.6407, 9.1592, 11.6989, 14.0608, 22.7236 ]

    create_LZ4_l1 = [ 0.0435, 0.7986, 2.3867, 8.5209, 18.8881, 25.9945, 35.0841, 45.7843, 54.8631, 67.5644, 79.7407, 90.9488, 105.7526, 121.2143, 134.6952, 161.6108, 185.0409 ]
    compute_LZ4_l1 = [ 0.0032, 0.0509, 0.1880, 0.7155, 1.6209, 2.2104, 2.9327, 5.1928, 6.0526, 7.4635, 8.9645, 10.5490, 12.0207, 13.7969, 15.8644, 19.2798, 21.3784 ]

    create_LZ4_l1_disk = [ 0.2557, 0.7487, 2.4254, 7.8367, 19.1367, 25.1097, 31.3328, 39.4257, 52.3823, 62.2994, 73.4805, 84.3078, 96.3005, 110.9688, 118.3864, 159.4544, 157.3727 ]
    compute_LZ4_l1_disk = [ 0.0037, 0.0590, 0.2268, 0.8837, 1.8008, 2.3744, 3.0909, 4.2624, 5.1138, 6.5483, 7.5345, 8.9750, 9.8907, 11.4285, 13.2415, 22.4300, 141.6707 ]

    create_ZSTD_l1 = [ 0.0423, 0.8595, 2.5674, 8.9603, 19.7700, 27.7205, 36.6830, 47.5384, 59.1740, 71.9198, 84.9254, 94.0010, 108.5841, 124.1261, 138.5614, 164.8593, 182.1642 ]
    compute_ZSTD_l1 = [ 0.0039, 0.0744, 0.2804, 1.0776, 2.3171, 3.4378, 4.6290, 6.7199, 8.3764, 9.3376, 11.0436, 12.8701, 15.1084, 17.1096, 19.1325, 23.3127, 25.9506 ]

    create_ZSTD_l1_disk = [ 0.1132, 0.7658, 2.5113, 8.0048, 19.8691, 26.8448, 35.4817, 43.4521, 58.6422, 64.7345, 75.8568, 85.5629, 99.6076, 114.3310, 121.0300, 158.5408, 161.0909 ]
    compute_ZSTD_l1_disk = [ 0.0043, 0.0813, 0.3313, 1.4464, 2.9211, 4.1365, 5.4587, 7.1266, 7.3236, 9.1663, 9.9776, 11.6081, 13.7075, 15.1375, 16.8231, 21.4002, 23.9236 ]

    create_numpy = [ 0.0020, 0.0550, 0.2232, 0.9468, 2.1856, 2.9516, 12.0596, 27.6355 ]
    compute_numpy = [ 0.0128, 0.3144, 1.3380, 5.5749, 38.6210, 70.7284, 164.0349, 325.4615 ]

    create_numpy_jit = [ 0.0024, 0.0603, 0.2329, 0.9657, 2.1673, 15.5171, 20.2344, 23.9815 ]
    compute_numpy_jit = [ 0.0050, 0.0393, 0.1333, 0.5318, 1.1473, 3.8321, 6.4264, 45.0717 ]

else:
    title_ = "Mac Mini M4 Pro (24 GB RAM)"

    create_numpy = [ 0.0016, 0.0415, 0.1631, 0.8974, 1.9819, 2.3129, 9.7300 ]
    compute_numpy = [ 0.0089, 0.2128, 0.9457, 5.7644, 36.5153, 63.8844, 137.9539 ]

    create_numpy_jit = [ 0.0018, 0.0436, 0.1676, 0.7349, 1.6885, 12.5894, 16.5044, 20.0384 ]
    compute_numpy_jit = [ 0.0038, 0.0205, 0.0642, 0.2606, 0.5486, 3.3116, 5.9220, 29.1374 ]

    create_l0 = [ 0.0344, 0.5770, 1.8655, 5.8634, 15.5161, 21.1114, 26.4065, 32.8173 ]
    compute_l0 = [ 0.0021, 0.0300, 0.0936, 0.3474, 0.7027, 8.4870, 11.1171, 31.2273 ]

    create_l0_disk = [ 0.0614, 0.5894, 1.9954, 6.4042, 16.9128, 21.5730, 26.9225, 33.8051, 45.1457, 53.1039, 63.7202, 69.6944, 79.1652 ]
    compute_l0_disk = [ 0.0027, 0.0427, 0.1650, 0.6768, 5.7428, 7.7228, 8.2640, 14.4505, 17.5742, 20.0730, 22.8288, 26.0431, 41.3722 ]

    create_BLOSCLZ_l7 = [ 0.0395, 0.5652, 1.9615, 5.8012, 15.8635, 18.7112, 23.2830, 29.0116, 43.6880, 49.6510, 59.9364, 65.2998, 75.2876, 92.7669, 372.2744, 119.3243, 117.3058 ]
    compute_BLOSCLZ_l7 = [ 0.0023, 0.0308, 0.1578, 0.3584, 1.3544, 1.0736, 1.3560, 1.7301, 3.7084, 4.4074, 5.4049, 6.1733, 4.5498, 4.9760, 5.4757, 6.4197, 6.9018 ]

    create_BLOSCLZ_l7_disk = [ 0.0422, 0.5557, 1.9601, 5.7647, 15.9145, 18.9607, 24.1283, 29.2553, 44.1869, 50.6621, 60.1618, 66.8329, 73.8509, 87.0546, 91.5202, 119.0131, 118.9790 ]
    compute_BLOSCLZ_l7_disk = [ 0.0022, 0.0313, 0.1729, 0.3894, 1.6717, 1.2707, 1.4595, 1.8445, 3.9138, 4.6782, 5.8595, 6.3338, 5.4898, 5.4879, 8.4475, 10.6740, 10.1856 ]

    create_BLOSCLZ_l9 = [ 0.0430, 0.6024, 1.9897, 5.8993, 15.7903, 20.1623, 24.1335, 29.4180, 43.8028, 50.2448, 60.9694, 65.2170, 69.7729, 88.4572, 90.5295, 119.4856, 119.4097 ]
    compute_BLOSCLZ_l9 = [ 0.0029, 0.0541, 0.1779, 0.3789, 1.4092, 1.9995, 1.4329, 1.8299, 3.9483, 4.6465, 5.6907, 6.4025, 4.5153, 8.4276, 5.9688, 6.7272, 7.8349 ]

    create_LZ4_l1 = [ 0.0361, 0.5804, 1.9389, 6.0536, 15.1991, 19.7225, 24.0663, 30.4482, 42.4730, 48.8970, 57.3124, 66.8990, 76.1380, 88.6604, 93.2565, 124.5175, 119.0430, 154.8972, 148.1766 ]
    compute_LZ4_l1 = [ 0.0021, 0.0303, 0.1018, 0.3595, 0.7678, 1.0191, 1.3130, 1.7165, 2.0468, 2.6400, 3.1438, 3.6971, 3.9760, 4.6626, 5.2315, 6.1437, 6.7120, 8.3231, 8.8490 ]

    create_LZ4_l1_disk = [ 0.1762, 0.5815, 1.9408, 6.6289, 16.4400, 20.2538, 25.0138, 31.3007, 43.0660, 49.9801, 58.6067, 67.7645, 77.3800, 89.2128, 95.8529, 126.9347, 122.4465 ]
    compute_LZ4_l1_disk = [ 0.0027, 0.0379, 0.1470, 0.5730, 1.0309, 1.3231, 1.7013, 2.6991, 3.0829, 3.7675, 4.2371, 4.9816, 5.3848, 6.0163, 6.8497, 12.3994, 12.0842 ]

    create_ZSTD_l1 = [ 0.0366, 0.5756, 1.9573, 6.1188, 15.5850, 19.9960, 24.9155, 30.7977, 42.7155, 49.7633, 58.7918, 67.7275, 77.1892, 88.9606, 116.8549, 180.0778, 140.9286, 209.7236, 1106.0708 ]
    compute_ZSTD_l1 = [ 0.0028, 0.0398, 0.1383, 0.5335, 1.0828, 1.6127, 2.2377, 2.7517, 3.2811, 4.3737, 4.6748, 5.3744, 6.2328, 6.6981, 9.7671, 12.4342, 29.5562, 37.8933, 19.2722 ]

    create_ZSTD_l1_disk = [ 0.1724, 0.6122, 2.0364, 6.4511, 16.3306, 20.9426, 25.9797, 32.1823, 45.2271, 51.2425, 59.8028, 68.1794, 78.3132, 90.4755, 96.8384, 129.1539, 125.2803 ]
    compute_ZSTD_l1_disk = [ 0.0030, 0.0452, 0.1687, 0.6854, 1.2524, 1.8355, 2.5684, 3.2852, 3.9175, 5.0215, 5.3327, 6.0550, 6.9507, 7.4801, 8.4181, 10.1903, 11.7509 ]

yaxis_title = 'Time (s)'
if iobw:
    yaxis_title = 'I/O bandwidth (GB/s)'
    # Convert times to I/O bandwidth
    create_l0 = sizes_GB[:len(create_l0)] / np.array(create_l0)
    compute_l0 = sizes_GB[:len(compute_l0)] / np.array(compute_l0)
    create_l0_disk = sizes_GB[:len(create_l0_disk)] / np.array(create_l0_disk)
    compute_l0_disk = sizes_GB[:len(compute_l0_disk)] / np.array(compute_l0_disk)
    create_BLOSCLZ_l7 = sizes_GB[:len(create_BLOSCLZ_l7)] / np.array(create_BLOSCLZ_l7)
    compute_BLOSCLZ_l7 = sizes_GB[:len(compute_BLOSCLZ_l7)] / np.array(compute_BLOSCLZ_l7)
    create_BLOSCLZ_l7_disk = sizes_GB[:len(create_BLOSCLZ_l7_disk)] / np.array(create_BLOSCLZ_l7_disk)
    compute_BLOSCLZ_l7_disk = sizes_GB[:len(compute_BLOSCLZ_l7_disk)] / np.array(compute_BLOSCLZ_l7_disk)
    create_LZ4_l1 = sizes_GB[:len(create_LZ4_l1)] / np.array(create_LZ4_l1)
    compute_LZ4_l1 = sizes_GB[:len(compute_LZ4_l1)] / np.array(compute_LZ4_l1)
    create_LZ4_l1_disk = sizes_GB[:len(create_LZ4_l1_disk)] / np.array(create_LZ4_l1_disk)
    compute_LZ4_l1_disk = sizes_GB[:len(compute_LZ4_l1_disk)] / np.array(compute_LZ4_l1_disk)
    create_ZSTD_l1 = sizes_GB[:len(create_ZSTD_l1)] / np.array(create_ZSTD_l1)
    compute_ZSTD_l1 = sizes_GB[:len(compute_ZSTD_l1)] / np.array(compute_ZSTD_l1)
    create_ZSTD_l1_disk = sizes_GB[:len(create_ZSTD_l1_disk)] / np.array(create_ZSTD_l1_disk)
    compute_ZSTD_l1_disk = sizes_GB[:len(compute_ZSTD_l1_disk)] / np.array(compute_ZSTD_l1_disk)
    create_numpy = sizes_GB[:len(create_numpy)] / np.array(create_numpy)
    compute_numpy = sizes_GB[:len(compute_numpy)] / np.array(compute_numpy)
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
        figure.add_annotation(x=64, y=y1_max * .9, text="64 GB", showarrow=True, arrowhead=2, ax=40, ay=0)
    elif m2linux:
        #y1_max = 100 if compute else y1_max
        figure.add_shape(
            type="line", x0=24, y0=0, x1=24, y1=y1_max,
            line=dict(color="Gray", width=2, dash="dot"),
        )
        figure.add_annotation(x=24, y=y1_max * .9, text="24 GB", showarrow=True, arrowhead=2, ax=40, ay=0)
    elif intel:
        #y1_max = 50 if compute else y1_max
        figure.add_shape(
            type="line", x0=32, y0=0, x1=32, y1=y1_max,
            line=dict(color="Gray", width=2, dash="dot"),
        )
        figure.add_annotation(x=32, y=y1_max * .9, text="32 GB", showarrow=True, arrowhead=2, ax=40, ay=0)
    else:
        #y1_max = 35 if compute else y1_max
        figure.add_shape(
            type="line", x0=24, y0=0, x1=24, y1=y1_max,
            line=dict(color="Gray", width=2, dash="dot"),
        )
        figure.add_annotation(x=24, y=y1_max * .9, text="24 GB", showarrow=True, arrowhead=2, ax=40, ay=0)

# Plot the data. There will be 2 plots: one for create times and another for compute times
labels = dict(
    l0="No compression", BLOSCLZ_l7="BLOSCLZ lvl=7", LZ4_l1="LZ4 lvl=1", ZSTD_l1="ZSTD lvl=1",
    numpy="NumPy", numpy_jit="NumPy (jit)"
)

if plotly:
    # Create the create times plot
    fig_create = go.Figure()
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_l0, mode='lines+markers', name=labels["l0"] + " (mem)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_l0_disk, mode='lines+markers', name=labels["l0"] + " (disk)"))
    # fig_create.add_trace(
    #     go.Scatter(x=sizes_GB, y=create_BLOSCLZ_l7, mode='lines+markers', name=labels["BLOSCLZ_l7"] + " (mem)"))
    # fig_create.add_trace(
    #     go.Scatter(x=sizes_GB, y=create_BLOSCLZ_l7_disk, mode='lines+markers', name=labels["BLOSCLZ_l7"] + " (disk)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_LZ4_l1, mode='lines+markers', name=labels["LZ4_l1"] + " (mem)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_LZ4_l1_disk, mode='lines+markers', name=labels["LZ4_l1"] + " (disk)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_ZSTD_l1, mode='lines+markers', name=labels["ZSTD_l1"] + " (mem)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_ZSTD_l1_disk, mode='lines+markers', name=labels["ZSTD_l1"] + " (disk)"))
    fig_create.add_trace(
        go.Scatter(x=sizes_GB, y=create_numpy_jit, mode='lines+markers',
                   name=labels["numpy"] + " (mem)", line=dict(color='brown')))
    fig_create.update_layout(title=f'Create operands: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title)

    # Add a vertical line at RAM limit
    add_ram_limit(fig_create, compute=False)

    # Create the compute times plot
    # Calculate the maximum y1 value
    y1_max = max(max(compute_l0), max(compute_l0_disk), max(compute_LZ4_l1), max(compute_LZ4_l1_disk),
                 max(compute_ZSTD_l1), max(compute_ZSTD_l1_disk), max(compute_numpy), max(compute_numpy_jit))

    fig_compute = go.Figure()
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_l0, mode='lines+markers', name=labels["l0"] + " (mem)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_l0_disk, mode='lines+markers', name=labels["l0"] + " (disk)"))
    # fig_compute.add_trace(
    #     go.Scatter(x=sizes_GB, y=compute_BLOSCLZ_l7, mode='lines+markers', name=labels["BLOSCLZ_l7"] + " (mem)"))
    # fig_compute.add_trace(
    #     go.Scatter(x=sizes_GB, y=compute_BLOSCLZ_l7_disk, mode='lines+markers', name=labels["BLOSCLZ_l7"] + " (disk)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_LZ4_l1, mode='lines+markers', name=labels["LZ4_l1"] + " (mem)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_LZ4_l1_disk, mode='lines+markers', name=labels["LZ4_l1"] + " (disk)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_ZSTD_l1, mode='lines+markers', name=labels["ZSTD_l1"] + " (mem)"))
    fig_compute.add_trace(
        go.Scatter(x=sizes_GB, y=compute_ZSTD_l1_disk, mode='lines+markers', name=labels["ZSTD_l1"] + " (disk)"))
    fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy, mode='lines+markers',
                                     name=labels["numpy"], line=dict(color='brown')))
    #fig_compute.add_trace(go.Scatter(x=sizes_GB, y=compute_numpy_jit, mode='lines+markers', name=labels["numpy_jit"]))
    fig_compute.update_layout(title=f'Blosc2 compute: {title_}', xaxis_title='Size (GB)', yaxis_title=yaxis_title)

    # Add a vertical line at RAM limit
    add_ram_limit(fig_compute, compute=True)

    # Show the plots
    fig_create.show()
    fig_compute.show()
else:
    plt.figure()
    plt.plot(sizes_GB, create_l0, "o-", label=labels["l0"])
    plt.plot(sizes_GB, create_LZ4_l1, "o-", label=labels["LZ4_l1"])
    plt.plot(sizes_GB, create_ZSTD_l1, "o-", label=labels["ZSTD_l1"])
    plt.plot(sizes_GB, create_numpy_jit, "o-", label=labels["numpy"])
    plt.xlabel("Size (GB)")
    plt.ylabel(yaxis_title)
    plt.title(f"Create operands ({title_})")
    plt.legend()
    # Now, the compute times
    plt.figure()
    plt.plot(sizes_GB, compute_l0, "o-", label=labels["l0"])
    plt.plot(sizes_GB, compute_LZ4_l1, "o-", label=labels["LZ4_l1"])
    plt.plot(sizes_GB, compute_ZSTD_l1, "o-", label=labels["ZSTD_l1"])
    plt.plot(sizes_GB, compute_numpy, "o-", label=labels["numpy"])
    #plt.plot(sizes_GB, compute_numpy_jit, "o-", label=labels["numpy_jit"])
    plt.xlabel("Size (GB)")
    plt.ylabel(yaxis_title)
    plt.title(f"Compute ({title_})")
    plt.legend()
    plt.show()
