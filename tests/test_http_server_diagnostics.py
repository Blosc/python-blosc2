"""HTTP request failures retain their exception line even when tests pass."""

import pytest

from conftest import report_http_server_error


@pytest.mark.parametrize("error", [ConnectionResetError("client disconnected"), ValueError("handler bug")])
def test_http_server_error_includes_summary_context_and_traceback(tmp_path, monkeypatch, capsys, error):
    log_path = tmp_path / "http-errors.log"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "example_test (call)")
    try:
        raise error
    except type(error):
        report_http_server_error(log_path, ("127.0.0.1", 12345))
    output = capsys.readouterr().err
    summary = f"{type(error).__name__}: {error}"
    assert output.startswith(f"HTTP request error: {summary}\n")
    assert "Test: example_test (call)" in output
    assert "Client: ('127.0.0.1', 12345)" in output
    assert "Traceback (most recent call last):" in output
    assert f"Full HTTP error log: {log_path}" in output
    saved = log_path.read_text(encoding="utf-8")
    assert saved in output
    assert saved.rstrip().endswith(summary)
