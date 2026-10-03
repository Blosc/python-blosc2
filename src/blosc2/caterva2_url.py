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


def service_probe_candidate(value):
    """Whether an HTTP URL is ambiguous rather than a recognized data source."""
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or "::" in parsed.path:
        return False
    suffixes = (
        ".b2nd",
        ".b2",
        ".b2t",
        ".b2o",
        ".b2z",
        ".b2d",
        ".b2f",
        ".b2frame",
        ".b2b",
        ".b2e",
        ".h5",
        ".hdf5",
        ".zarr",
        ".parquet",
    )
    return not any(
        decode_service_component(part).lower().endswith(suffixes) for part in parsed.path.split("/")
    )


def validate_roots(roots):
    """Validate the Caterva2 roots mapping without assuming a root name prefix."""
    if not isinstance(roots, dict) or len(roots) > 1000:
        raise ValueError("Invalid Caterva2 roots response")
    for name, metadata in roots.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or any(c in name for c in "/\\:%?#")
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or not isinstance(metadata, dict)
            or metadata.get("name", name) != name
        ):
            raise ValueError("Invalid Caterva2 root entry")
    return roots


def discover_service(value, *, required=False, auth_token=None):
    """Probe api/roots with bounded bytes, time, and same-origin redirects.

    Return a validated roots mapping or None for a conclusively non-service
    response. Authentication, connectivity, and server failures are preserved.
    """
    import time

    from blosc2.c2array import _auth_headers, _server_url, _sync_client

    parsed = validate_service_url(value)
    url = _server_url(value.rstrip("/"), "api/roots")
    client = _sync_client()
    deadline = time.monotonic() + 3
    for _ in range(4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Caterva2 service discovery exceeded 3 seconds")
        with client.stream(
            "GET", url, headers=_auth_headers(auth_token), timeout=remaining, follow_redirects=False
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                from urllib.parse import urljoin

                target = urljoin(url, response.headers.get("location", ""))
                redirect = urlsplit(target)
                if (
                    redirect.scheme,
                    redirect.hostname,
                    redirect.port or (443 if redirect.scheme == "https" else 80),
                ) != (
                    parsed.scheme,
                    parsed.hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                ):
                    raise ValueError("Caterva2 discovery cannot redirect credentials to another origin")
                if redirect.username is not None or redirect.password is not None:
                    raise ValueError("Unsafe Caterva2 discovery redirect")
                url = target
                continue
            if response.status_code in {404, 405}:
                if required:
                    response.raise_for_status()
                return None
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise TimeoutError("Caterva2 service discovery exceeded 3 seconds")
                payload.extend(chunk)
                if len(payload) > 1 << 20:
                    raise ValueError("Caterva2 roots response exceeds 1 MiB")
            import json

            try:
                return validate_roots(json.loads(payload))
            except (ValueError, TypeError) as error:
                if required:
                    raise ValueError("Invalid Caterva2 roots response") from error
                return None
    raise ValueError("Too many Caterva2 discovery redirects")
