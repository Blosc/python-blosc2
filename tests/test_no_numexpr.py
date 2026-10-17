"""NumExpr installation must not affect runtime import or expression dispatch."""

import json
import subprocess
import sys
import textwrap


def test_runtime_never_imports_numexpr():
    source = textwrap.dedent("""
        import sys, json, importlib.abc
        sys.path[:] = json.loads(sys.argv[1])
        class BlockNumExpr(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numexpr" or fullname.startswith("numexpr."):
                    raise AssertionError("Unexpected NumExpr runtime import")
        sys.meta_path.insert(0, BlockNumExpr())
        import numpy as np
        import blosc2
        x = np.arange(24.).reshape(4,6)
        for order in ("C", "F"):
            result=blosc2.evaluate("x * 2 + 1", local_dict={"x":x}, order=order)
            np.testing.assert_array_equal(result,x*2+1)
        stored=blosc2.asarray(x)
        np.testing.assert_array_equal(blosc2.evaluate("x + 1",local_dict={"x":stored}),x+1)
        out=np.empty(x.shape,dtype="complex128")
        assert blosc2.evaluate("complex(x,x)",local_dict={"x":x},out=out) is out
        np.testing.assert_array_equal(out.real,x)
        evaluator=blosc2.prepare_evaluate("x+1")
        np.testing.assert_array_equal(evaluator({"x":x}),x+1)
        blosc2.evaluate("sum(x)",local_dict={"x":x})
        assert blosc2.re_evaluate({"x":x}) == x.sum()
        assert "numexpr" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", source, json.dumps(sys.path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
