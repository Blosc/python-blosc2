#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Unified roofline plotter for different machines and disk/memory modes.
# The user selects the benchmark via `machine` and `mem_mode` below.

import matplotlib.pyplot as plt
import ast

# ---------------------------------------------------------------------
# User selection
# ---------------------------------------------------------------------
# Valid machines: "Apple-M4-Pro", "AMD-7800X3D"
# machine = "Apple-M4-Pro"
machine = "AMD-7800X3D"
# False -> on-disk benchmark, True -> in-memory benchmark
mem_mode = False

# ---------------------------------------------------------------------
# Benchmark dictionaries (raw string form, as produced by driver script)
# ---------------------------------------------------------------------

BENCH_DATA = {
    "Apple-M4-Pro": {
        "disk": """
{'blosc2': {'low': {'GFLOPS': 2.570591026389536,
                    'Intensity': 5.5,
                    'Time': 28.884407997131348},
            'matmul0': {'GFLOPS': 46.26183975097429,
                        'Intensity': 1000,
                        'Time': 0.04863619804382324},
            'matmul1': {'GFLOPS': 438.1365321396617,
                        'Intensity': 5000,
                        'Time': 0.641923189163208},
            'matmul2': {'GFLOPS': 448.8428100084526,
                        'Intensity': 10000,
                        'Time': 5.012890815734863},
            'medium': {'GFLOPS': 14.146962346220464,
                       'Intensity': 36.75,
                       'Time': 35.06936597824097},
            'very low': {'GFLOPS': 0.49123569734016437,
                         'Intensity': 0.5,
                         'Time': 13.74085807800293}},
 'blosc2-nocomp': {'low': {'GFLOPS': 0.03860960944488331,
                           'Intensity': 5.5,
                           'Time': 1923.0963759422302},
                   'matmul0': {'GFLOPS': 32.9184188862999,
                               'Intensity': 1000,
                               'Time': 0.06835079193115234},
                   'matmul1': {'GFLOPS': 375.8405170559847,
                               'Intensity': 5000,
                               'Time': 0.7483227252960205},
                   'matmul2': {'GFLOPS': 399.46900484462606,
                               'Intensity': 10000,
                               'Time': 5.632477045059204},
                   'medium': {'GFLOPS': 0.46027450974226586,
                              'Intensity': 36.75,
                              'Time': 1077.8893671035767},
                   'very low': {'GFLOPS': 0.006658136735463883,
                                'Intensity': 0.5,
                                'Time': 1013.7971429824829}},
 'numpy/numexpr': {'low': {'GFLOPS': 0.03342497696428004,
                           'Intensity': 5.5,
                           'Time': 2221.3927052021027},
                   'matmul0': {'GFLOPS': 3.6124326198946726,
                               'Intensity': 1000,
                               'Time': 0.6228489875793457},
                   'matmul1': {'GFLOPS': 93.36108303946814,
                               'Intensity': 5000,
                               'Time': 3.0124971866607666},
                   'matmul2': {'GFLOPS': 277.86243889802796,
                               'Intensity': 10000,
                               'Time': 8.097532033920288},
                   'medium': {'GFLOPS': 0.09460263438020816,
                              'Intensity': 36.75,
                              'Time': 5244.3042759895325},
                   'very low': {'GFLOPS': 0.0015092629683608571,
                                'Intensity': 0.5,
                                'Time': 4472.381646871567}}}
""",
        "mem": """
{'blosc2': {'low': {'GFLOPS': 3.2804978086093888,
                    'Intensity': 5.5,
                    'Time': 0.7544586658477783},
            'matmul0': {'GFLOPS': 104.37977259655798,
                        'Intensity': 1000,
                        'Time': 0.02155590057373047},
            'matmul1': {'GFLOPS': 542.7544356959245,
                        'Intensity': 5000,
                        'Time': 0.5181901454925537},
            'matmul2': {'GFLOPS': 550.8998283178123,
                        'Intensity': 10000,
                        'Time': 4.084227085113525},
            'medium': {'GFLOPS': 24.37674704003205,
                       'Intensity': 36.75,
                       'Time': 0.678412914276123},
            'very low': {'GFLOPS': 0.9103679794411528,
                         'Intensity': 0.5,
                         'Time': 0.24715280532836914}},
 'blosc2-nocomp': {'low': {'GFLOPS': 2.745232662043899,
                           'Intensity': 5.5,
                           'Time': 0.9015629291534424},
                   'matmul0': {'GFLOPS': 75.94463400502156,
                               'Intensity': 1000,
                               'Time': 0.029626846313476562},
                   'matmul1': {'GFLOPS': 505.49157655447544,
                               'Intensity': 5000,
                               'Time': 0.5563890933990479},
                   'matmul2': {'GFLOPS': 516.0177547765433,
                               'Intensity': 10000,
                               'Time': 4.3603150844573975},
                   'medium': {'GFLOPS': 22.45272072521166,
                              'Intensity': 36.75,
                              'Time': 0.7365477085113525},
                   'very low': {'GFLOPS': 0.5840329482970421,
                                'Intensity': 0.5,
                                'Time': 0.3852522373199463}},
 'numpy/numexpr': {'low': {'GFLOPS': 5.746789246798714,
                           'Intensity': 5.5,
                           'Time': 0.4306752681732178},
                   'matmul0': {'GFLOPS': 666.4677966101694,
                               'Intensity': 1000,
                               'Time': 0.003376007080078125},
                   'matmul1': {'GFLOPS': 945.7058955100038,
                               'Intensity': 5000,
                               'Time': 0.2973968982696533},
                   'matmul2': {'GFLOPS': 974.8577951206411,
                               'Intensity': 10000,
                               'Time': 2.3080289363861084},
                   'medium': {'GFLOPS': 29.044906245027512,
                              'Intensity': 36.75,
                              'Time': 0.5693769454956055},
                   'very low': {'GFLOPS': 1.5056997530170846,
                                'Intensity': 0.5,
                                'Time': 0.14943218231201172}}}
"""
    },
    "AMD-7800X3D": {
        "disk": """
{'blosc2': {'low': {'GFLOPS': 2.6569613592385535,
                    'Intensity': 5.5,
                    'Time': 27.945457220077515},
            'matmul0': {'GFLOPS': 12.553085867977686,
                        'Intensity': 1000,
                        'Time': 0.17923879623413086},
            'matmul1': {'GFLOPS': 240.360991381506,
                        'Intensity': 5000,
                        'Time': 1.1701149940490723},
            'matmul2': {'GFLOPS': 268.0288488506098,
                        'Intensity': 10000,
                        'Time': 8.39461874961853},
            'medium': {'GFLOPS': 15.532085276343903,
                       'Intensity': 36.75,
                       'Time': 31.941944122314453},
            'very low': {'GFLOPS': 0.5656500608225292,
                         'Intensity': 0.5,
                         'Time': 11.933172941207886}},
 'blosc2-nocomp': {'low': {'GFLOPS': 1.0313162899034,
                           'Intensity': 5.5,
                           'Time': 71.99537205696106},
                   'matmul0': {'GFLOPS': 14.36429529261525,
                               'Intensity': 1000,
                               'Time': 0.15663838386535645},
                   'matmul1': {'GFLOPS': 215.303286764059,
                               'Intensity': 5000,
                               'Time': 1.3062968254089355},
                   'matmul2': {'GFLOPS': 273.333776088537,
                               'Intensity': 10000,
                               'Time': 8.231693983078003},
                   'medium': {'GFLOPS': 6.643671590137467,
                              'Intensity': 36.75,
                              'Time': 74.67632818222046},
                   'very low': {'GFLOPS': 0.12206790616761651,
                                'Intensity': 0.5,
                                'Time': 55.29709005355835}},
 'numpy/numexpr': {'low': {'GFLOPS': 1.357592296775474,
                           'Intensity': 5.5,
                           'Time': 54.69241404533386},
                   'matmul0': {'GFLOPS': 14.61036282906348,
                               'Intensity': 1000,
                               'Time': 0.15400028228759766},
                   'matmul1': {'GFLOPS': 219.1569896084874,
                               'Intensity': 5000,
                               'Time': 1.2833266258239746},
                   'matmul2': {'GFLOPS': 309.16178854453585,
                               'Intensity': 10000,
                               'Time': 7.277742862701416},
                   'medium': {'GFLOPS': 7.66225952699885,
                              'Intensity': 36.75,
                              'Time': 64.74917721748352},
                   'very low': {'GFLOPS': 0.18572341000005319,
                                'Intensity': 0.5,
                                'Time': 36.34436821937561}}}
""",
        "mem": """
{'blosc2': {'low': {'GFLOPS': 2.2049809120053325,
                    'Intensity': 5.5,
                    'Time': 1.1224586963653564},
            'matmul0': {'GFLOPS': 71.74383457503421,
                        'Intensity': 1000,
                        'Time': 0.03136157989501953},
            'matmul1': {'GFLOPS': 265.6029172803062,
                        'Intensity': 5000,
                        'Time': 1.0589115619659424},
            'matmul2': {'GFLOPS': 297.90536239084577,
                        'Intensity': 10000,
                        'Time': 7.552734136581421},
            'medium': {'GFLOPS': 12.334163526222097,
                       'Intensity': 36.75,
                       'Time': 1.3407881259918213},
            'very low': {'GFLOPS': 0.4098550921015945,
                         'Intensity': 0.5,
                         'Time': 0.5489745140075684}},
 'blosc2-nocomp': {'low': {'GFLOPS': 1.9901502643717384,
                           'Intensity': 5.5,
                           'Time': 1.2436246871948242},
                   'matmul0': {'GFLOPS': 55.69960455645399,
                               'Intensity': 1000,
                               'Time': 0.040395259857177734},
                   'matmul1': {'GFLOPS': 267.0038256315959,
                               'Intensity': 5000,
                               'Time': 1.0533556938171387},
                   'matmul2': {'GFLOPS': 302.88209627168624,
                               'Intensity': 10000,
                               'Time': 7.428633213043213},
                   'medium': {'GFLOPS': 11.669410440193081,
                              'Intensity': 36.75,
                              'Time': 1.4171667098999023},
                   'very low': {'GFLOPS': 0.38086456224635085,
                                'Intensity': 0.5,
                                'Time': 0.5907611846923828}},
 'numpy/numexpr': {'low': {'GFLOPS': 4.547634034022808,
                           'Intensity': 5.5,
                           'Time': 0.5442390441894531},
                   'matmul0': {'GFLOPS': 272.5225677900026,
                               'Intensity': 1000,
                               'Time': 0.008256196975708008},
                   'matmul1': {'GFLOPS': 363.40324566643244,
                               'Intensity': 5000,
                               'Time': 0.7739336490631104},
                   'matmul2': {'GFLOPS': 369.9673735674775,
                               'Intensity': 10000,
                               'Time': 6.08161735534668},
                   'medium': {'GFLOPS': 17.90938592011286,
                              'Intensity': 36.75,
                              'Time': 0.923398494720459},
                   'very low': {'GFLOPS': 1.5235763064852037,
                                'Intensity': 0.5,
                                'Time': 0.14767885208129883}}}
"""
    },
}

# ---------------------------------------------------------------------
# Select benchmark
# ---------------------------------------------------------------------
mode_key = "mem" if mem_mode else "disk"
try:
    result_str = BENCH_DATA[machine][mode_key]
except KeyError as e:
    raise SystemExit(f"Unknown selection: machine={machine!r}, mem_mode={mem_mode}") from e

legend = "in-memory" if mem_mode else "on-disk"

# Parse the result string as a dictionary
results = ast.literal_eval(result_str)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

styles = {
    'numpy/numexpr': {'color': 'blue', 'marker': 'o', 'label': 'NumPy/NumExpr'},
    'blosc2': {'color': 'red', 'marker': 's', 'label': 'Blosc2 (compressed)'},
    'blosc2-nocomp': {'color': 'green', 'marker': '^', 'label': 'Blosc2 (uncompressed)'},
}

# Plot each backend's results
for backend, backend_results in results.items():
    intensities = []
    gflops = []
    labels = []
    for workload, metrics in backend_results.items():
        intensities.append(metrics['Intensity'])
        gflops.append(metrics['GFLOPS'])
        labels.append(workload)

    style = styles[backend]
    ax.loglog(
        intensities,
        gflops,
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
        markersize=8,
        linestyle='',
        alpha=0.7,
    )

# Build a single annotation per unique x (Intensity)
intensity_map = {}
for backend_results in results.values():
    for workload, metrics in backend_results.items():
        intensity = metrics['Intensity']
        gflop = metrics['GFLOPS']
        if intensity not in intensity_map:
            intensity_map[intensity] = {'label': workload, 'gflops': []}
        intensity_map[intensity]['gflops'].append(gflop)

# Axes limits
ax.set_xlim(0.1, 5e4)
ymin = 0.1 if mem_mode else 0.001
ax.set_ylim(ymin, 2000.0)

# Annotate once per intensity, centered under the cluster of points
for intensity, info in sorted(intensity_map.items()):
    raw_ypos = min(info['gflops']) * 0.6
    ymin_curr, ymax_curr = ax.get_ylim()
    safe_ypos = max(raw_ypos, ymin_curr * 1.5 if ymin_curr > 0 else raw_ypos)
    ax.annotate(
        info['label'],
        (intensity, safe_ypos),
        ha='center',
        va='top',
        fontsize=10,
        alpha=0.9,
    )

ax.set_xlabel('Arithmetic Intensity (FLOPs/element)', fontsize=12)
ax.set_ylabel('Performance (GFLOPS/sec)', fontsize=12)
machine2 = machine.replace("-", " ")
ax.set_title(f'Roofline Analysis: {machine2} ({legend})', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(False)

plt.tight_layout()
plt.savefig(f'roofline_plot-{machine}-{legend}.png', dpi=300, bbox_inches='tight')
plt.show()
