# Menudet draft: numeric certification

Menudet requires checked integer results and the stated floating-point type,
domain, NaN, infinity, signed-zero and rounding rules. It does **not** promise
bit-identical transcendental results across host math libraries or a universal
ULP error bound. Native exact-anchor and exceptional-value tests remain separate
from the bounded independent math check described here.

## Reproducible independent references

`scripts/menudet_math_vectors.json` contains 284 samples for 40 canonical
operations, covering both float32 and float64. Inputs and expected results are
stored as exact hexadecimal encodings. References were generated with mpmath
1.3.0 at both 100 and 200 decimal digits; regeneration rejects disagreement
between the two rounded results. The checker does not need mpmath and does not
use NumPy or the executing host's libm as its mathematical reference.

The finite domains include inverse-function endpoints, small arguments that
exercise cancellation, mixed quadrants, gamma functions, and explicit fused
multiply-add cancellation. Domain errors, NaNs/infinities, subnormal behavior,
signed zeros, rounding functions, nextafter and aliases additionally have native
conformance fixtures; they are not certified merely by these finite samples.

| Operations | Maximum error for the checked samples |
| --- | --- |
| cospi, sinpi, exp10, lgamma, tgamma, logaddexp | 8 ULP |
| Other sampled transcendental functions, sqrt, hypot, floating pow | 4 ULP |
| fdim, fmin, fmax, copysign, fmod, remainder, ldexp, fma | Exact rounded value |

These are **function-specific acceptance budgets for this finite corpus**, not
new normative bounds over all possible inputs. ULP is measured using the smaller
adjacent gap at the expected value, including at powers of two. Adding a new
sample or changing a budget requires review; a failed platform must not silently
skip a function or inflate its tolerance.

## Running certification

Against an installed package:

```console
python scripts/certify_menudet_math.py --report menudet-math.json
```

Reports identify the package path/version, platform, reference generator, sample
count, per-function/per-precision maxima and individual failures. The process
exits nonzero on failure. Use `--generate` only to regenerate the checked-in
reference corpus in a development environment with mpmath installed.

Native Python CI uploads the reports. The same corpus runs in Pyodide and in the
installed-wheel test command, without adding a runtime reference-library
dependency. Passing a local wheel certifies only that tested wheel and host;
release readiness additionally requires the supported platform matrix to pass.
