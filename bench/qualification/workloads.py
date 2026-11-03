"""Small, deterministic resident-map contracts; no numerical backend matrix."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    case_id: str
    recipe: str
    backend: str | None
    explicit_output: bool
    jit: bool = True
    workload_revision: int = 1
    frontend: str = "evaluate"
    storage: str = "numpy"
    compressed_output: bool = False
    math_backend: str = "system"
    route: str = "resident"
    fixed_size: int | None = None

    @property
    def expression(self):
        return {
            "comparison": "x < y",
            "polynomial": "(x + y) * (x - y)",
            "square": "x**2 + y**2",
            "transcendental": "sin(x) < cos(y)",
            "loop": "x + y * 8",
        }[self.recipe]

    def reference(self, inputs):
        x, y = inputs["x"], inputs["y"]
        if self.recipe == "transcendental":
            return np.sin(x) < np.cos(y)
        if self.recipe == "square":
            return x**2 + y**2
        if self.recipe == "loop":
            result = x.copy()
            for _ in range(8):
                result = result + y
            return result
        return x < y if self.recipe == "comparison" else (x + y) * (x - y)

    def shape(self, size):
        return (max(1, size // 256), 256) if self.storage == "slabs" else (size,)

    def grid(self, size):
        return ((64, 256), (16, 256)) if self.storage == "slabs" else ((16384,), (4096,))

    def bind(self, size, seed):
        rng = np.random.default_rng(seed)
        # Exactly representable inputs and arithmetic: exact references even at
        # comparison boundaries, with no implicit power/precision assumptions.
        x, y = (rng.integers(-32, 33, size=size).astype("float64") / 8 for _ in range(2))
        # Keep transcendental comparisons away from equality, with both outcomes.
        if self.recipe == "transcendental":
            y[:] = 2
        x, y = (
            x[: np.prod(self.shape(size))].reshape(self.shape(size)),
            y[: np.prod(self.shape(size))].reshape(self.shape(size)),
        )
        if self.storage == "strided":
            x, y = np.repeat(x, 2)[::2], np.repeat(y, 2)[::2]
        expected = self.reference({"x": x, "y": y})
        destination = np.empty_like(expected) if self.explicit_output else None
        if self.storage in ("compressed", "slabs"):
            import blosc2

            chunks, blocks = self.grid(size)
            x, y = (
                blosc2.asarray(
                    a, chunks=chunks, blocks=blocks, cparams={"codec": blosc2.Codec.LZ4, "clevel": 1}
                )
                for a in (x, y)
            )
        return {"x": x, "y": y}, expected, destination

    def prepare(self, inputs, destination):
        """Return a callable and its diagnostic owner, without evaluating values."""
        import blosc2

        if self.frontend == "evaluate":
            return lambda: blosc2.evaluate(
                self.expression, local_dict=inputs, out=destination, optimization="aggressive"
            ), None
        if self.frontend == "lazyexpr":
            lazy = blosc2.lazyexpr(self.expression, inputs)
            chunks, blocks = self.grid(inputs["x"].size)
            return lambda: lazy.compute(
                chunks=chunks, blocks=blocks, cparams={"codec": blosc2.Codec.LZ4, "clevel": 1}
            ), lazy
        from blosc2.dsl_kernel import SourceKernel

        source = (
            "def k(x, y):\n    result = x\n    for i in range(8):\n        result = result + y\n    return result\n"
            if self.recipe == "loop"
            else f"def k(x, y):\n    return {self.expression}\n"
        )
        output_dtype = "bool" if self.recipe in ("comparison", "transcendental") else "float64"
        artifact = SourceKernel.from_source(source).export(
            {"x": "float64", "y": "float64"}, output_dtype, version="0.1"
        )
        kernel = blosc2.MenudetKernel.from_json(artifact, jit=self.jit)
        if self.storage == "numpy":
            return lambda: kernel.evaluate(inputs), kernel
        lazy = kernel.lazy(inputs)
        if self.compressed_output:
            return lambda: lazy.compute(cparams={"codec": blosc2.Codec.LZ4, "clevel": 1}), lazy
        return lambda: lazy[:], lazy

    def metadata(self, size, seed):
        return {
            **asdict(self),
            "expression": self.expression,
            "shape": list(self.shape(size)),
            "input_dtype": "float64",
            "output_dtype": "bool" if self.recipe in ("comparison", "transcendental") else "float64",
            "seed": seed,
            "allocation": "compressed"
            if self.compressed_output
            else "explicit-numpy"
            if self.explicit_output
            else "fresh-numpy",
            "storage_grid": self.grid(size) if self.storage in ("compressed", "slabs") else None,
            "codec": "LZ4" if self.storage in ("compressed", "slabs") else None,
            "clevel": 1 if self.storage in ("compressed", "slabs") else None,
            "bytes_processed": int(np.prod(self.shape(size)))
            * (16 + (1 if self.recipe in ("comparison", "transcendental") else 8)),
            "requested_policy": {
                "jit": self.jit,
                "jit_backend": self.backend,
                "math_backend": self.math_backend,
            },
        }


CASES = {
    case.case_id: case
    for recipe in ("comparison", "polynomial")
    for backend in (None, "cc", "tcc")
    for explicit in (False, True)
    for case in [
        Case(
            f"evaluate-{recipe}-{backend or 'auto'}-{'out' if explicit else 'new'}",
            recipe,
            backend,
            explicit,
        )
    ]
}

for case in [
    Case(
        "authored-slabs-compressed",
        "comparison",
        "cc",
        False,
        frontend="authored",
        storage="slabs",
        compressed_output=True,
        route="menudet-prefilter",
    ),
    Case(
        "authored-slabs-numpy",
        "comparison",
        "cc",
        False,
        frontend="authored",
        storage="slabs",
        route="menudet-prefilter",
    ),
    Case(
        "lazyexpr-slabs-compressed",
        "polynomial",
        "cc",
        False,
        frontend="lazyexpr",
        storage="slabs",
        compressed_output=True,
        route="menudet-prefilter",
    ),
    Case(
        "authored-loop-interpreter",
        "loop",
        "cc",
        False,
        jit=False,
        frontend="authored",
        route="authored-flat",
        fixed_size=16384,
    ),
    Case(
        "authored-loop-cc", "loop", "cc", False, frontend="authored", route="authored-flat", fixed_size=16384
    ),
    Case("evaluate-transcendental-system", "transcendental", "cc", False),
    Case(
        "evaluate-transcendental-accelerate",
        "transcendental",
        "cc",
        False,
        jit=False,
        math_backend="accelerate",
    ),
    Case("evaluate-square-accelerate", "square", "cc", False, math_backend="accelerate"),
    Case(
        "lazyexpr-tiny-compressed",
        "polynomial",
        "cc",
        False,
        frontend="lazyexpr",
        storage="compressed",
        compressed_output=True,
        route="fallback",
        fixed_size=1024,
    ),
    Case(
        "lazyexpr-above-compressed",
        "polynomial",
        "cc",
        False,
        frontend="lazyexpr",
        storage="compressed",
        compressed_output=True,
        route="menudet-prefilter",
        fixed_size=65536,
    ),
    Case(
        "evaluate-strided-fallback",
        "polynomial",
        "cc",
        False,
        storage="strided",
        route="fallback",
        fixed_size=4096,
    ),
]:
    CASES[case.case_id] = case


def validate_result(result, expected, destination, compressed=False):
    if compressed:
        import blosc2

        if type(result) is not blosc2.NDArray:
            raise AssertionError("Expected compressed result")
        result = result[:]
    if type(result) is not np.ndarray or result.shape != expected.shape or result.dtype != expected.dtype:
        raise AssertionError("Output type, shape or dtype contract failed")
    if destination is not None and result is not destination:
        raise AssertionError("Explicit output identity contract failed")
    np.testing.assert_array_equal(result, expected)
