.. _title:

.. title:: Python-Blosc2 Documentation

.. raw:: html

    <p style="text-align: center; color: black; background-color: rgba(230, 169, 9, 0.65);">
        <a href="https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md"
           style="font-size: 1.5em;">Version 4.12.0 released on 2026-09-01!</a>
        <span style="display: inline-block; width: 20px;"></span>
        <span style="font-family: monospace;">pip install blosc2 -U</span>
    </p>

.. raw:: html

    <h1 class="text-center">Python-Blosc2: <em>Compress Better, Compute Bigger</em></h1>

.. raw:: html

    <!-- Does not look good on phones
    .. image:: https://github.com/Blosc/python-blosc2/blob/main/images/b2nd-2level-parts.png?raw=true
      :width: 25%
      :align: center
    -->

.. grid:: 1 2 3 3
    :gutter: 3

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-compress" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Top-Notch Compression</h3>
            </div>

        `Combine advanced codecs and filters <https://www.blosc.org/docs/2025-EuroSciPy-Blosc2.pdf>`_ for efficient `lossless <https://www.blosc.org/posts/bytedelta-enhance-compression-toolset/>`_ and `lossy <https://www.blosc.org/posts/blosc2-lossy-compression/>`_ compression to reduce storage space while maintaining high performance.

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-cubes" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Full-Fledged NDArrays</h3>
            </div>

        `NDArray objects <https://www.blosc.org/python-blosc2/tutorials/01.ndarray-basics.html>`_ enable efficient storage and manipulation of arbitrarily large N-dimensional datasets, following the `Array API <https://data-apis.org/array-api/latest/>`_ standard, with an additional `C API <https://www.blosc.org/posts/blosc2-ndim-intro/>`_.

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-bolt" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Compute Engine Inside</h3>
            </div>

        Combines compression with high-speed computation of complex `mathematical expressions <https://ironarray.io/blog/blosc2-eval-expressions>`_ and `reductions <https://www.blosc.org/python-blosc2/tutorials/04.reductions.html>`_, while maintaining compatibility with NumPy.

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-sitemap" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Hierarchical Structures</h3>
            </div>

        Efficiently store data hierarchically with the `TreeStore class <https://www.blosc.org/python-blosc2/reference/tree_store.html#blosc2.TreeStore>`_ for convenience and optimized `performance <https://www.blosc.org/posts/new-treestore-blosc2/>`_.

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-hdd" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Flexible Storage</h3>
            </div>

        Access data from anywhere: read/write in `memory or disk <https://www.blosc.org/docs/Exploring-MilkyWay-SciPy2023-paper.pdf>`_, stream from `the network <https://www.blosc.org/python-blosc2/reference/c2array.html>`_, or use `memory-mapped files <https://www.blosc.org/python-blosc2/reference/storage.html#blosc2.Storage.params.mmap_mode>`_ for high-performance I/O.

    .. grid-item-card::
        :class-card: intro-card text-center no-border

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-file-code" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Uncomplicated Format</h3>
            </div>

        `Blosc2's format <https://github.com/Blosc/c-blosc2/blob/main/README_FORMAT.rst>`_ is simple and accessible, with specifications under 4000 words that make it easy to read and integrate.


.. raw:: html

    <h1 class="text-center">Documentation</h1>

.. grid:: 1 2 3 3
    :gutter: 3

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-rocket" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Getting Started</h3>
            </div>

        New to Python-Blosc2? Start here for installation instructions and a high-level overview of core concepts.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="getting_started/index.html" class="btn btn-info">To getting started</a>
            </div>

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-graduation-cap" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Tutorials</h3>
            </div>

        Step-by-step interactive tutorials walking you through NDArrays, lazy expressions, reductions, SChunk, CTable, and more.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="tutorials/index.html" class="btn btn-info">To the tutorials</a>
            </div>

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-lightbulb" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Guides</h3>
            </div>

        Task-oriented recipes and in-depth guides on performance optimization, remote arrays, IPC sharing, and CLI tools.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="guides/index.html" class="btn btn-info">To the guides</a>
            </div>

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-book" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">API Reference</h3>
            </div>

        The reference guide provides a comprehensive description of the Python-Blosc2 API, detailing functions and parameters.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="reference/index.html" class="btn btn-info">To the reference guide</a>
            </div>

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-code" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Development</h3>
            </div>

        Found a typo in the documentation or want to improve functionality? The contributing guidelines will walk you through the process.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="development/index.html" class="btn btn-info">To the development guide</a>
            </div>

    .. grid-item-card::
        :class-card: intro-card text-center

        .. raw:: html

            <div style="text-align: center;">
                <i class="fas fa-newspaper" style="font-size: 2em; color: #007acc; margin-bottom: 10px;"></i>
                <h3 style="margin-top: 10px; margin-bottom: 15px;">Release Notes</h3>
            </div>

        Want to see what's new in the latest release? Explore the release notes to discover new features, improvements, and bug fixes.

        .. raw:: html

            <div style="text-align: center; margin-top: 20px;">
                <a href="release_notes/index.html" class="btn btn-info">To the release notes</a>
            </div>



..  toctree::
    :maxdepth: 1
    :hidden:

    Getting Started <getting_started/index>
    Tutorials <tutorials/index>
    Guides <guides/index>
    API Reference <reference/index>
    Development <development/index>
    Release Notes <release_notes/index>
