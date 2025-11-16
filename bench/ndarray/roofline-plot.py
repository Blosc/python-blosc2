#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This plots the roofline results from the 7800X3D CPU when using
# the flops-vs-arithmetic-intensity.py script.  The default is potting
# the disk results.  To plot the in-memory results, set mem_mode=True
# below.

import matplotlib.pyplot as plt
import ast

mem_mode = False
legend = "on-disk"

result = """
{'blosc2': {'low': {'GFLOPS': 6.327852773687432,
                    'Intensity': 11.0,
                    'Time': 78.22558736801147},
            'matmul0': {'GFLOPS': 139.6079603241232,
                        'Intensity': 2000,
                        'Time': 0.12893247604370117},
            'matmul1': {'GFLOPS': 285.5730560584079,
                        'Intensity': 10000,
                        'Time': 7.878894567489624},
            'matmul2': {'GFLOPS': 275.25046270149494,
                        'Intensity': 20000,
                        'Time': 65.39498543739319},
            'medium': {'GFLOPS': 41.090081101862665,
                       'Intensity': 73.5,
                       'Time': 80.49387860298157},
            'very low': {'GFLOPS': 1.8312096896017473,
                         'Intensity': 0.5,
                         'Time': 12.286959886550903}},
 'blosc2-nocomp': {'low': {'GFLOPS': 2.411916710121926,
                           'Intensity': 11.0,
                           'Time': 205.23096752166748},
                   'matmul0': {'GFLOPS': 1.3107291477971774,
                               'Intensity': 2000,
                               'Time': 13.732814311981201},
                   'matmul1': {'GFLOPS': 250.65634985918948,
                               'Intensity': 10000,
                               'Time': 8.976433277130127},
                   'matmul2': {'GFLOPS': 259.70444823051554,
                               'Intensity': 20000,
                               'Time': 69.30955600738525},
                   'medium': {'GFLOPS': 16.14344559666435,
                              'Intensity': 73.5,
                              'Time': 204.88191199302673},
                   'very low': {'GFLOPS': 0.17230079619691702,
                                'Intensity': 0.5,
                                'Time': 130.58558344841003}},
 'numpy/numexpr': {'low': {'GFLOPS': 3.507544548886978,
                           'Intensity': 11.0,
                           'Time': 141.1243658065796},
                   'matmul0': {'GFLOPS': 0.9969869122268231,
                               'Intensity': 2000,
                               'Time': 18.054399490356445},
                   'matmul1': {'GFLOPS': 285.9938969965044,
                               'Intensity': 10000,
                               'Time': 7.867300748825073},
                   'matmul2': {'GFLOPS': 331.3412081775571,
                               'Intensity': 20000,
                               'Time': 54.32466459274292},
                   'medium': {'GFLOPS': 20.830858799336102,
                              'Intensity': 73.5,
                              'Time': 158.77885937690735},
                   'very low': {'GFLOPS': 0.2501104809770888,
                                'Intensity': 0.5,
                                'Time': 89.96024441719055}}}
"""

result_mem = """
{'blosc2': {'low': {'GFLOPS': 4.866522333848727,
                    'Intensity': 11.0,
                    'Time': 2.5428836345672607},
            'matmul0': {'GFLOPS': 85.0395948600573,
                        'Intensity': 1000,
                        'Time': 0.026458263397216797},
            'matmul1': {'GFLOPS': 268.91185402574024,
                        'Intensity': 5000,
                        'Time': 1.045881748199463},
            'matmul2': {'GFLOPS': 299.895418397909,
                        'Intensity': 10000,
                        'Time': 7.502615451812744},
            'medium': {'GFLOPS': 35.53410004647491,
                       'Intensity': 73.5,
                       'Time': 2.3269901275634766},
            'very low': {'GFLOPS': 0.9519585532368179,
                         'Intensity': 0.5,
                         'Time': 0.5908870697021484}},
 'blosc2-nocomp': {'low': {'GFLOPS': 4.6922734647067745,
                           'Intensity': 11.0,
                           'Time': 2.6373143196105957},
                   'matmul0': {'GFLOPS': 58.503040710180954,
                               'Intensity': 1000,
                               'Time': 0.03845953941345215},
                   'matmul1': {'GFLOPS': 273.2579472297275,
                               'Intensity': 5000,
                               'Time': 1.0292472839355469},
                   'matmul2': {'GFLOPS': 301.0879153190238,
                               'Intensity': 10000,
                               'Time': 7.472900390625},
                   'medium': {'GFLOPS': 32.61464583666784,
                              'Intensity': 73.5,
                              'Time': 2.535287380218506},
                   'very low': {'GFLOPS': 0.6691234986315745,
                                'Intensity': 0.5,
                                'Time': 0.8406519889831543}},
 'numpy/numexpr': {'low': {'GFLOPS': 9.115128287351304,
                           'Intensity': 11.0,
                           'Time': 1.357633113861084},
                   'matmul0': {'GFLOPS': 266.15855825365935,
                               'Intensity': 1000,
                               'Time': 0.008453607559204102},
                   'matmul1': {'GFLOPS': 362.16460995052523,
                               'Intensity': 5000,
                               'Time': 0.7765805721282959},
                   'matmul2': {'GFLOPS': 371.33744743751464,
                               'Intensity': 10000,
                               'Time': 6.059178829193115},
                   'medium': {'GFLOPS': 46.40747408437248,
                              'Intensity': 73.5,
                              'Time': 1.781771183013916},
                   'very low': {'GFLOPS': 2.681374682770323,
                                'Intensity': 0.5,
                                'Time': 0.20978045463562012}}}
"""

if mem_mode:
    result = result_mem
    legend = "in-memory"

# Parse the result string as a dictionary
results = ast.literal_eval(result)

# Create roofline plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors and markers for each backend
styles = {
    'numpy/numexpr': {'color': 'blue', 'marker': 'o', 'label': 'NumPy/NumExpr'},
    'blosc2': {'color': 'red', 'marker': 's', 'label': 'Blosc2 (compressed)'},
    'blosc2-nocomp': {'color': 'green', 'marker': '^', 'label': 'Blosc2 (uncompressed)'}
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
    ax.loglog(intensities, gflops,
              marker=style['marker'],
              color=style['color'],
              label=style['label'],
              markersize=8,
              linestyle='',
              alpha=0.7)

# Build a single annotation per unique x (Intensity)
intensity_map = {}
for backend_results in results.values():
    for workload, metrics in backend_results.items():
        intensity = metrics['Intensity']
        gflop = metrics['GFLOPS']
        # Apply same mapping used above, but only for annotation text
        if intensity not in intensity_map:
            intensity_map[intensity] = {'label': workload, 'gflops': []}
        intensity_map[intensity]['gflops'].append(gflop)

# Ensure labels fit inside the plot by extending the lower y-limit if necessary
all_min_gflops = [min(info['gflops']) for info in intensity_map.values()]
cur_ymin, cur_ymax = ax.get_ylim()
new_ymin = min(cur_ymin, min(all_min_gflops) * 0.5)
ax.set_ylim(new_ymin, cur_ymax)

# Annotate once per intensity, centered under the cluster of points
for intensity, info in sorted(intensity_map.items()):
    # place label below the lowest point for this x, but keep a margin above x-axis
    raw_ypos = min(info['gflops']) * 0.6
    ymin, ymax = ax.get_ylim()
    # ensure at least 1.5x above current ymin to avoid overlapping the axis
    safe_ypos = max(raw_ypos, ymin * 1.5 if ymin > 0 else raw_ypos)
    ax.annotate(info['label'],
                (intensity, safe_ypos),
                ha='center',
                va='top',
                fontsize=10,
                alpha=0.9)

ax.set_xlabel('Arithmetic Intensity (FLOPs/element)', fontsize=12)
ax.set_ylabel('Performance (GFLOPS/sec)', fontsize=12)
ax.set_title(f'Roofline Analysis: AMD 7800X3D ({legend})', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(False)

plt.tight_layout()
plt.savefig(f'roofline_plot-7800X3D-{legend}.png', dpi=300, bbox_inches='tight')
plt.show()
