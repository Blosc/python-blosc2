"""Recognition of Caterva2 service URLs, separate from source format detection."""

from urllib.parse import unquote, urlsplit, urlunsplit

import blosc2


def validate_service_url(value):
    """Validate a service URL without silently normalizing unsafe components."""
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ValueError("Caterva2 service URLs require http:// or https://")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value) or "\\" in value:
        raise ValueError("Unsafe Caterva2 service URL")
    parsed = urlsplit(value)
    if (
        not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Caterva2 service URL requires an authority without userinfo")
    _ = parsed.port
    if parsed.query or parsed.fragment or "::" in parsed.path:
        raise ValueError("Caterva2 service URLs do not support queries, fragments, or selectors")
    for component in parsed.path.split("/"):
        decoded = decode_service_component(component)
        if (
            decoded in {".", ".."}
            or any(c in decoded for c in "/\\%?#")
            or any(ord(c) < 32 or ord(c) == 127 for c in decoded)
        ):
            raise ValueError("Unsafe Caterva2 URL path component")
    return parsed


def decode_service_component(component):
    """Decode exactly once, rejecting malformed escapes and invalid UTF-8."""
    import re

    if re.search(r"%(?![0-9a-fA-F]{2})", component):
        raise ValueError("Malformed URL escape")
    return unquote(component, errors="strict")


def caterva2_urlpath(value):
    """Return a URLPath for an @-root service URL, or None for an ordinary URL.

    This function never makes a network request. Use an explicit fsspec override
    when an ordinary HTTP source happens to have an @-prefixed path component.
    """
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return None
    parsed = urlsplit(value)
    components = parsed.path.split("/")
    marker = next(
        (i for i, component in enumerate(components) if decode_service_component(component).startswith("@")),
        None,
    )
    if marker is None:
        return None
    validate_service_url(value)
    logical = [decode_service_component(component) for component in components[marker:]]
    if logical[-1] == "":
        logical.pop()
    if not logical or logical[0] == "@" or any(not component for component in logical):
        raise ValueError("Caterva2 URL requires a root and nonempty dataset components")
    base = urlunsplit((parsed.scheme, parsed.netloc, "/".join(components[:marker]), "", ""))
    return blosc2.URLPath("/".join(logical), urlbase=base)
