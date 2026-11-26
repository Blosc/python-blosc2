"""Generate the migration fixture using the released blosc2==4.16.1 wheel.

Run in an empty working directory. Keep the two output files together.
"""

import numpy as np

import blosc2


def main():
    if blosc2.__version__ != "4.16.1":
        raise RuntimeError(f"Expected the 4.16.1 wheel, got {blosc2.__version__}")
    gain, bias = np.float64(2), 1.0

    @blosc2.menudet(jit=False)
    def cubic(x):
        return (x * gain + bias) ** 3

    x = blosc2.asarray(np.array([0, 1, 2, 3], dtype="float32"), urlpath="x.b2nd", mode="w")
    lazy = cubic.specialize(x).lazy({"x": x}, partitions=(2,))
    lazy.save("kernel.b2nd", mode="w")
    payload = lazy.schunk.vlmeta["b2o"]
    assert payload["version"] == 1
    assert payload["operands"]["x"]["urlpath"] == "x.b2nd"


if __name__ == "__main__":
    main()
