"""Bounded startup and continuously drained output for gateway acceptance tests."""

import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from queue import Empty, Queue


@contextmanager
def gateway_process(command, timeout=20):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events = Queue()
    logs = deque(maxlen=20)

    def drain(stream, name):
        for line in stream:
            logs.append(f"{name}: {line[-2000:].rstrip()}")
            if name == "stdout" and line.startswith("listening on "):
                events.put(("ready", line.removeprefix("listening on ").strip()))
        events.put(("closed", name))

    threads = [
        threading.Thread(target=drain, args=(stream, name), daemon=True)
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
    ]
    try:
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.start()
        while True:
            try:
                event, value = events.get(timeout=max(0, deadline - time.monotonic()))
            except Empty:
                raise TimeoutError("cat2lite startup timed out\n" + "\n".join(logs)) from None
            if event == "ready":
                yield value if value.startswith("http") else "http://" + value
                break
            if value == "stdout":
                raise RuntimeError("cat2lite exited before listening\n" + "\n".join(logs))
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        for stream in (process.stdout, process.stderr):
            stream.close()
