#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import io
from textwrap import TextWrapper

# print_dict is nice, but rich is way more popular and can be used for more things
# import print_dict
import rich


def info_text_report_(items: list) -> str:
    with io.StringIO() as buf:
        print(items, file=buf)
        return buf.getvalue()


def info_text_report(items: list) -> str:
    keys = [k for k, v in items]
    max_key_len = max(len(k) for k in keys)
    report = ""
    for k, v in items:
        if type(v) is dict:
            # print_dict way
            # text = k.ljust(max_key_len) + " : " + print_dict.format_dict(v, sort_keys=True)
            # rich way
            with io.StringIO() as buf:
                v_sorted = {k: val for k, val in sorted(v.items())}
                rich.print(v_sorted, file=buf)
                str_v = buf.getvalue()[:-1]  # remove the trailing \n
            text = k.ljust(max_key_len) + " : " + str_v
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
        report += (
            "<tr>"
            '<th style="text-align: left">%s</th>'
            '<td style="text-align: left">%s</td>'
            "</tr>" % (k, v)
        )
    report += "</tbody>"
    report += "</table>"
    return report


class InfoReporter(object):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        items = self.obj.info_items
        return info_text_report(items)

    def _repr_html_(self):
        items = self.obj.info_items
        return info_html_report(items)
