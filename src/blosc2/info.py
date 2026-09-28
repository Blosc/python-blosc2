#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import io
import pprint
from html import escape
from textwrap import TextWrapper


def format_nbytes_human(nbytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{nbytes} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return None


def format_nbytes_info(nbytes: int) -> str:
    return f"{nbytes} ({format_nbytes_human(nbytes)})"


def format_store_tree(entries: dict[str, str], root: str = "/") -> str:
    """Render relative paths and optional node labels without opening their values."""
    children = {}
    for path in entries:
        if not path.strip("/"):
            continue
        parent = ""
        for name in path.strip("/").split("/"):
            child = f"{parent}/{name}".strip("/")
            children.setdefault(parent, set()).add(child)
            parent = child
    lines = [root]
    pending = [("", "")]
    while pending:
        path, prefix = pending.pop()
        if path:
            lines.append(prefix + path.rsplit("/", 1)[-1] + entries.get(path, ""))
            prefix = prefix[:-4] + ("    " if prefix.endswith("└── ") else "│   ")
        names = sorted(children.get(path, ()))
        pending.extend(
            (name, prefix + ("└── " if i == len(names) - 1 else "├── "))
            for i, name in reversed(list(enumerate(names)))
        )
    return "\n".join(lines)


def info_text_report_(items: list) -> str:
    with io.StringIO() as buf:
        print(items, file=buf)
        return buf.getvalue()


def info_text_report(items: list) -> str:
    keys = [k for k, v in items]
    max_key_len = max(len(k) for k in keys)
    report = ""
    for k, v in items:
        if isinstance(v, str) and "\n" in v:
            text = k.ljust(max_key_len) + " : " + v.replace("\n", "\n" + " " * (max_key_len + 3))
        elif isinstance(v, dict):
            # rich way, this is disabled because it doesn't work well in the notebooks
            # with io.StringIO() as buf:
            #     v_sorted = {k: val for k, val in sorted(v.items())}
            #     rich.print(v_sorted, file=buf)
            #     str_v = buf.getvalue()[:-1]  # remove the trailing \n
            # text = k.ljust(max_key_len) + " : " + str_v
            # pprint way
            text = k.ljust(max_key_len) + " : " + pprint.pformat(v)
        else:
            wrapper = TextWrapper(
                width=96,
                initial_indent=k.ljust(max_key_len) + " : ",
                subsequent_indent=" " * max_key_len + " : ",
            )
            text = wrapper.fill(str(v))
        report += text + "\n"
    return report


def info_html_report(items: list) -> str:
    report = '<table class="NDArray-info">'
    report += "<tbody>"
    for k, v in items:
        value = escape(str(v))
        if isinstance(v, str) and "\n" in v:
            value = f"<pre>{value}</pre>"
        report += (
            f'<tr><th style="text-align: left">{escape(k)}</th>'
            f'<td style="text-align: left">{value}</td></tr>'
        )
    report += "</tbody>"
    report += "</table>"
    return report


class InfoReporter:
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        items = self.obj.info_items
        return info_text_report(items)

    def _repr_html_(self):
        items = self.obj.info_items
        return info_html_report(items)
