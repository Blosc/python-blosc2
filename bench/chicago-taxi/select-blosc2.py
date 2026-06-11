import sys
import time

import blosc2


blosc2.set_printoptions(display_index=True)

columns = ["payment.tips", "payment.total", "trip.sec", "trip.km", "company"]
path = sys.argv[1] if len(sys.argv) > 1 else "chicago-taxi.b2z"

t0 = time.perf_counter()
with blosc2.open(path) as t:
#with blosc2.open(path, mmap_mode="r") as t:
    t1 = time.perf_counter()

    condition = (t.payment.tips > 100) & (t.trip.km > 0) & (t.trip.begin.lon < 0)
    result = t.where(condition, columns=columns).sort_by("trip.sec")
    t2 = time.perf_counter()

    print(result)
    t3 = time.perf_counter()

    print(f"open:    {t1 - t0:.6f} s")
    print(f"compute: {t2 - t1:.6f} s")
    print(f"print:   {t3 - t2:.6f} s")
    print(f"total:   {t3 - t0:.6f} s")
