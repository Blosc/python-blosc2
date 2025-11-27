#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This script compares the performance impact of different DDR5 memory speeds
# (4800 MT/s vs 6000 MT/s) on NumPy/NumExpr operations on an AMD 7800X3D system.
# It plots GFLOPS vs Arithmetic Intensity to visualize how memory bandwidth
# affects performance across different workload intensities.

mem_4800 = {'low': {'GFLOPS': 4.493354439009314,
                    'Intensity': 5.5,
                    'Time': 0.5508134365081787},
            'matmul0': {'GFLOPS': 258.19222456293943,
                        'Intensity': 1000,
                        'Time': 0.008714437484741211},
            'matmul1': {'GFLOPS': 364.1837565094117,
                        'Intensity': 5000,
                        'Time': 0.7722749710083008},
            'matmul2': {'GFLOPS': 370.6084229401238,
                        'Intensity': 10000,
                        'Time': 6.0710978507995605},
            'medium': {'GFLOPS': 17.71942775308632,
                       'Intensity': 36.75,
                       'Time': 0.9332976341247559},
            'very low': {'GFLOPS': 1.0880454532877077,
                         'Intensity': 0.5,
                         'Time': 0.20679283142089844}
            }

mem_6000 = {'low': {'GFLOPS': 4.530616712594456,
                    'Intensity': 5.5,
                    'Time': 0.5462832450866699},
            'matmul0': {'GFLOPS': 241.78069276491084,
                        'Intensity': 1000,
                        'Time': 0.009305953979492188},
            'matmul1': {'GFLOPS': 364.46651669646604,
                        'Intensity': 5000,
                        'Time': 0.7716758251190186},
            'matmul2': {'GFLOPS': 371.2794341995866,
                        'Intensity': 10000,
                        'Time': 6.0601255893707275},
            'medium': {'GFLOPS': 17.79626768253134,
                       'Intensity': 36.75,
                       'Time': 0.9292678833007812},
            'very low': {'GFLOPS': 1.4817325114381805,
                         'Intensity': 0.5,
                         'Time': 0.15184926986694336}
            }

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Collect intensities and GFLOPS for each memory speed
    def extract_xy(mem_dict):
        intensities, gflops = [], []
        for name, metrics in mem_dict.items():
            intensities.append(metrics["Intensity"])
            gflops.append(metrics["GFLOPS"])
        # Sort by intensity for nicer lines
        order = sorted(range(len(intensities)), key=lambda i: intensities[i])
        intensities = [intensities[i] for i in order]
        gflops = [gflops[i] for i in order]
        return intensities, gflops

    x4800, y4800 = extract_xy(mem_4800)
    x6000, y6000 = extract_xy(mem_6000)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot performance curves for both memory speeds
    ax.loglog(x4800, y4800, "-o", label="DDR5 @ 4800 MT/s", alpha=0.8)
    ax.loglog(x6000, y6000, "-s", label="DDR5 @ 6000 MT/s", alpha=0.8)

    # Same limits as roofline-plot2.py for mem_mode=True
    ax.set_xlim(0.1, 5e4)
    ax.set_ylim(0.1, 2000.0)

    # Annotate the first data point where the performance difference is most visible
    # (memory-bound region shows the biggest impact of faster RAM)
    x0_4800, y0_4800 = x4800[0], y4800[0]
    x0_6000, y0_6000 = x6000[0], y6000[0]

    # 6000 has larger value, annotate above with more spacing
    ax.annotate(
        f"{y0_6000:.2f} GFLOPS",
        (x0_6000, y0_6000),
        xytext=(x0_6000 * 2.5, y0_6000 * 3.5),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=9,
        ha="left",
        va="bottom",
    )

    # 4800 has smaller value, annotate below
    ax.annotate(
        f"{y0_4800:.2f} GFLOPS",
        (x0_4800, y0_4800),
        xytext=(x0_4800 * 2.5, y0_4800 * 0.55),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=9,
        ha="left",
        va="top",
    )

    # --- single workload label per workload name (avoid duplicates) ---
    # Build a map: workload name -> list of (intensity, gflops) across mem_4800/mem_6000
    workload_map: dict[str, dict[str, list[float]]] = {}

    for workload, metrics in mem_4800.items():
        intensity = metrics["Intensity"]
        gflops = metrics["GFLOPS"]
        if workload not in workload_map:
            workload_map[workload] = {"intensity": [], "gflops": []}
        workload_map[workload]["intensity"].append(intensity)
        workload_map[workload]["gflops"].append(gflops)

    for workload, metrics in mem_6000.items():
        intensity = metrics["Intensity"]
        gflops = metrics["GFLOPS"]
        if workload not in workload_map:
            workload_map[workload] = {"intensity": [], "gflops": []}
        workload_map[workload]["intensity"].append(intensity)
        workload_map[workload]["gflops"].append(gflops)

    # Place a single label per workload at the average intensity and slightly below
    # the minimum GFLOPS across both memory speeds for that workload.
    for workload, vals in workload_map.items():
        intensities = vals["intensity"]
        gflops_list = vals["gflops"]
        x_label = sum(intensities) / len(intensities)
        y_min = min(gflops_list)
        raw_ypos = y_min * 0.6

        ymin_curr, _ = ax.get_ylim()
        safe_ypos = max(raw_ypos, ymin_curr * 1.5 if ymin_curr > 0 else raw_ypos)

        # Avoid overlap between matmul1 and matmul2 by using different vertical offsets
        if workload == "matmul1":
            safe_ypos *= .8   # push matmul1 a bit higher
        elif workload == "matmul2":
            safe_ypos *= 1.2   # keep matmul2 lower

        ax.annotate(
            workload,
            (x_label, safe_ypos),
            ha="center",
            va="top",
            fontsize=10,
            alpha=0.9,
        )
    # --------------------------------------------------------------

    ax.set_xlabel("Arithmetic Intensity (FLOPs/element)")
    ax.set_ylabel("Performance (GFLOPS/sec)")
    ax.set_title("Memory speed impact on NumPy/NumExpr performance\nAMD 7800X3D (in-memory)")
    ax.legend(loc="upper left")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("roofline-mem-speed-AMD-7800X3D.png", dpi=300, bbox_inches="tight")
    plt.show()
