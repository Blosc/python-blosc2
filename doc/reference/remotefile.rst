.. _RemoteFile:

RemoteFile
==========

``RemoteFile`` is a read-only Caterva2 fixed-chunk SChunk byte stream. It is
returned by lazy service leaf opening or ``RemoteStore`` file lookup. Discovery
and file metadata do not fetch payload. Ordinary files such as Markdown, JPEG,
and PDF published as compressed byte streams are supported; there is no Python
object deserialization or implicit interpretation as an array.

.. code-block:: python

    import blosc2

    with blosc2.open("https://cat2.cloud/demo/@public/examples/README.md") as file:
        prefix = file.read_bytes(0, min(file.nbytes, 4096))
        file.download("README.md")

``read_bytes(start, stop)`` returns original bytes, with at most 16 MiB per call.
The omitted stop means the file end, not an unbounded streaming read. Use
``download`` for larger files. A chunk can contain more data than the requested
range: transfers are chunk-granular, bounded at 32 MiB compressed and 256 MiB
decoded per chunk. Oversized/irregular streams require rechunking on the server.
Generic fixed-chunk typed SChunks expose their raw byte representation; they do
not imply a text/image/document format.

Downloads stage beside the destination, then publish completed original bytes.
Existing destinations are not overwritten unless ``overwrite=True`` is explicit.
No-overwrite publication uses a hard link to prevent concurrent-creation races;
the destination filesystem must support hard links. ``progress(done, total)``
and ``cancel()`` callbacks run on the calling thread. Cancellation or failure
removes only the operation's staging file and preserves any previous destination.

Cache policy/allowance, traffic, and lifetime are shared with the source owner.
``cache_bytes`` includes other leaves in that owner. MEMORY/DISK retain compressed
chunks; NONE retains no payload between calls. Transient decoded buffers are
separately bounded, not included in retained cache bytes. Sources must remain
immutable until explicit refresh; returned children outlive their parent handle.
``save`` of a file reference is intentionally unsupported; ``download`` is an
original-byte export, not a reference archive. Explicit nonlazy ``URLPath`` input
still requires ``lazy=True`` for byte-stream access.

.. autoclass:: blosc2.RemoteFile
    :members: read_bytes, download, close, source, name, media_type, nbytes, cbytes, nchunks, chunksize, attrs, traffic, info, cache_policy, max_cache_bytes, cache_bytes, metadata_bytes
