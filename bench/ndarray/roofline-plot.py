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
machine = "AMD-7800X3D"
# False -> on-disk benchmark, True -> in-memory benchmark
mem_mode = False

# ---------------------------------------------------------------------
# Benchmark dictionaries (raw string form, as produced by driver script)
# ---------------------------------------------------------------------

BENCH_DATA = {
    "Apple-M4-Pro": {
        "disk": """
{'blosc2': {'low': {'GFLOPS': 3.3687118577656445,
                    'Intensity': 11.0,
                    'Time': 146.94043922424316},
            'matmul0': {'GFLOPS': 204.23876446288816,
                        'Intensity': 2000,
                        'Time': 0.08813214302062988},
            'matmul1': {'GFLOPS': 529.6595422477051,
                        'Intensity': 10000,
                        'Time': 4.248011827468872},
            'matmul2': {'GFLOPS': 344.9923084930752,
                        'Intensity': 20000,
                        'Time': 52.1750762462616},
            'medium': {'GFLOPS': 3.2360774704788646,
                       'Intensity': 73.5,
                       'Time': 1022.0707106590271},
            'very low': {'GFLOPS': 0.8304666278735933,
                         'Intensity': 0.5,
                         'Time': 27.093201875686646}},
 'blosc2-nocomp': {'low': {'GFLOPS': 0.3274009717467319,
                           'Intensity': 11.0,
                           'Time': 1511.9075467586517},
                   'matmul0': {'GFLOPS': 38.83544895025038,
                               'Intensity': 2000,
                               'Time': 0.46349406242370605},
                   'matmul1': {'GFLOPS': 452.8072228948917,
                               'Intensity': 10000,
                               'Time': 4.96900200843811},
                   'matmul2': {'GFLOPS': 339.7051311123031,
                               'Intensity': 20000,
                               'Time': 52.9871301651001},
                   'medium': {'GFLOPS': 1.558588531217065,
                              'Intensity': 73.5,
                              'Time': 2122.112368822098},
                   'very low': {'GFLOPS': 0.005054729090605573,
                                'Intensity': 0.5,
                                'Time': 4451.277130126953}},
 'numpy/numexpr': {'low': {'GFLOPS': 0.09951421858654366,
                           'Intensity': 11.0,
                           'Time': 4974.1635620594025},
                   'matmul0': {'GFLOPS': 0.7754047828096021,
                               'Intensity': 2000,
                               'Time': 23.213681936264038},
                   'matmul1': {'GFLOPS': 519.4085605579936,
                               'Intensity': 10000,
                               'Time': 4.331850051879883},
                   'matmul2': {'GFLOPS': 761.113465433182,
                               'Intensity': 20000,
                               'Time': 23.649561882019043},
                   'medium': {'GFLOPS': 0.22064810795082324,
                              'Intensity': 73.5,
                              'Time': 14989.93139219284},
                   'very low': {'GFLOPS': 0.003910290255103866,
                                'Intensity': 0.5,
                                'Time': 5754.048557043076}}}
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
            'matmul0': {'GFLOPS': 110.16077007145368,
                        'Intensity': 2000,
                        'Time': 0.16339755058288574},
            'matmul1': {'GFLOPS': 278.1667526617581,
                        'Intensity': 10000,
                        'Time': 8.08867335319519},
            'matmul2': {'GFLOPS': 254.47585605595123,
                        'Intensity': 20000,
                        'Time': 70.73362588882446},
            'medium': {'GFLOPS': 15.532085276343903,
                       'Intensity': 36.75,
                       'Time': 31.941944122314453},
            'very low': {'GFLOPS': 0.5656500608225292,
                         'Intensity': 0.5,
                         'Time': 11.933172941207886}},
 'blosc2-nocomp': {'low': {'GFLOPS': 1.0313162899034,
                           'Intensity': 5.5,
                           'Time': 71.99537205696106},
                   'matmul0': {'GFLOPS': 5.026447790269603,
                               'Intensity': 2000,
                               'Time': 3.5810577869415283},
                   'matmul1': {'GFLOPS': 240.53695304016009,
                               'Intensity': 10000,
                               'Time': 9.354072093963623},
                   'matmul2': {'GFLOPS': 244.4603185993202,
                               'Intensity': 20000,
                               'Time': 73.63158202171326},
                   'medium': {'GFLOPS': 6.643671590137467,
                              'Intensity': 36.75,
                              'Time': 74.67632818222046},
                   'very low': {'GFLOPS': 0.12206790616761651,
                                'Intensity': 0.5,
                                'Time': 55.29709005355835}},
 'numpy/numexpr': {'low': {'GFLOPS': 1.357592296775474,
                           'Intensity': 5.5,
                           'Time': 54.69241404533386},
                   'matmul0': {'GFLOPS': 2.7506948651842946,
                               'Intensity': 2000,
                               'Time': 6.5438010692596436},
                   'matmul1': {'GFLOPS': 275.4348725971461,
                               'Intensity': 10000,
                               'Time': 8.16890025138855},
                   'matmul2': {'GFLOPS': 342.9817247403082,
                               'Intensity': 20000,
                               'Time': 52.48093032836914},
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
