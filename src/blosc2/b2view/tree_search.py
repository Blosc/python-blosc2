"""Listing-only searches shared by all b2view tree backends."""

from collections import deque
from dataclasses import dataclass, field

MAX_DIRECTORIES = 200
MAX_NODES = 10_000
MAX_DEPTH = 12


def matching_nodes(listings, query):
    """Case-insensitive path matches over discovered nodes; performs no I/O."""
    query = query.casefold().strip()
    nodes = {node.path: node for children in listings.values() for node in children}
    return sorted((node for node in nodes.values() if query in node.path.casefold()), key=lambda n: n.path)


@dataclass
class SearchReport:
    listings: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    directories: int = 0
    status: str = "Complete"


def _enqueue_children(queue, children, depth, max_depth):
    groups = [node.path for node in children if node.has_children]
    if depth >= max_depth:
        return bool(groups)
    queue.extend((path, depth + 1) for path in groups)
    return False


def search_tree(
    browser,
    known,
    *,
    cancel,
    progress=None,
    max_directories=MAX_DIRECTORIES,
    max_nodes=MAX_NODES,
    max_depth=MAX_DEPTH,
):
    """Breadth-first discovery without previewing files, arrays or tables.

    Known listings are reused. Limits are checked between directory listings;
    a backend's final indivisible listing may exceed the node budget.
    """
    report = SearchReport(dict(known))
    queue = deque([("/", 0)])
    visited = set()
    depth_limited = False
    while queue:
        if cancel():
            report.status = "Cancelled"
            break
        path, depth = queue.popleft()
        if path in visited:
            continue
        if report.directories >= max_directories:
            report.status = "Directory limit reached"
            break
        visited.add(path)
        report.directories += 1
        try:
            if path not in report.listings:
                with browser.io_lock:
                    if cancel():
                        report.status = "Cancelled"
                        break
                    children = browser.list_children(path)
                if cancel():
                    report.status = "Cancelled"
                    break
                report.listings[path] = children
            children = report.listings[path]
            if progress is not None:
                progress(report)
            if len({node.path for values in report.listings.values() for node in values}) >= max_nodes:
                report.status = "Node limit reached"
                break
            depth_limited |= _enqueue_children(queue, children, depth, max_depth)
        except Exception as error:
            report.errors.append((path, error))
    else:
        if depth_limited:
            report.status = "Depth limit reached"
        elif report.errors:
            report.status = "Completed with listing errors"
    return report
