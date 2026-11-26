"""Compare the transition JSON preflight with the legacy native reader."""

import json
from pathlib import Path

import pytest

import blosc2
from blosc2.menudet_legacy import decode_legacy_artifact


@pytest.fixture
def released_json():
    if not blosc2.blosc2_ext.menudet_artifact_available():
        pytest.skip("Differential checks require the legacy native adapter")
    fixture = Path(__file__).parent / "data" / "legacy_menudet_v1" / "kernel.b2nd"
    return blosc2.open(fixture, mode="r").kernel.to_json()


def test_python_preflight_preserves_released_manifest(released_json):
    assert decode_legacy_artifact(released_json.encode()) == json.loads(released_json)


@pytest.mark.parametrize(
    "replacement",
    [
        b'"metadata":{"nested":{"a":1,"a":2}}',
        b'"metadata":{"value":"\\u0000"}',
        b'"metadata":{"value":"\\ud800"}',
        b'"metadata":{"value":NaN}',
        b'"metadata":[]',
    ],
)
def test_legacy_json_metadata_rejections(released_json, replacement):
    data = released_json.rstrip().encode()[:-1] + b"," + replacement + b"}"
    with pytest.raises(blosc2.MenudetError):
        decode_legacy_artifact(data)
    with pytest.raises(blosc2.MenudetError):
        blosc2.MenudetKernel.from_json(data, jit=False)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("language", "version", "0.2"),
        ("output", "contract", "array"),
        ("output", "dtype", "auto"),
        ("context", "ndim", True),
        ("semantics", "integer_powers", "pow"),
        ("semantics", "casting", "trunc"),
        ("semantics", "fp", "fast"),
    ],
)
def test_legacy_manifest_rejections(released_json, section, key, value):
    manifest = json.loads(released_json)
    if key == "fp":
        # Otherwise the released producer's flexible-math capability may opt in.
        manifest["requires"] = ["numeric"]
    manifest[section][key] = value
    data = json.dumps(manifest).encode()
    with pytest.raises(blosc2.MenudetError):
        decode_legacy_artifact(data)
    with pytest.raises(blosc2.MenudetError):
        blosc2.MenudetKernel.from_json(data, jit=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [("category", "unknown"), ("encoding", "decimal"), ("value", "3FF0000000000000")],
)
def test_legacy_capture_rejections(released_json, key, value):
    manifest = json.loads(released_json)
    manifest["constants"][0][key] = value
    data = json.dumps(manifest).encode()
    with pytest.raises(blosc2.MenudetError):
        decode_legacy_artifact(data)
    with pytest.raises(blosc2.MenudetError):
        blosc2.MenudetKernel.from_json(data, jit=False)


def test_legacy_envelope_does_not_substitute_for_source_checks(released_json):
    manifest = json.loads(released_json)
    header, result = manifest["source"].split("\n    return ", 1)
    manifest["source"] = f"{header}\n    if x > 0:\n        return {result}\n    return {result}\n"
    manifest["requires"].remove("control-flow")
    data = json.dumps(manifest).encode()
    assert decode_legacy_artifact(data) == manifest
    with pytest.raises(blosc2.MenudetError, match="source capability"):
        blosc2.MenudetKernel.from_json(data, jit=False)
