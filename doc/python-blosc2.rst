.. _title:

.. title:: Python-Blosc2 Documentation

.. raw:: html

    <p style="text-align: center; color: black; background-color: rgba(230, 169, 9, 0.65);">
        <a href="https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md"
           style="font-size: 1.5em;">Version 3.5.0 released on 2025-06-24!</a>
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

.. panels::
    :card: intro-card text-center no-border
    :column: col-lg-4 col-md-6 col-sm-12 mb-4 d-flex
    :container: + gap-3

    **Excellent compression**

    `Combine advanced codecs and filters <https://blosc.org/docs/LEAPS-INNOV_WP7_D74_v1.pdf>`_ for efficient `lossless <https://www.blosc.org/posts/bytedelta-enhance-compression-toolset/>`_ and `lossy <https://www.blosc.org/posts/blosc2-lossy-compression/>`_ compression, reducing storage space while keeping high performance.

    ---

    **Compressed NDArrays**

    `NDArray objects <https://www.blosc.org/python-blosc2/getting_started/tutorials/01.ndarray-basics.html>`_ allow for efficient storage and manipulation of arbitrarily large N-dim datasets. A `C-API <https://www.blosc.org/posts/blosc2-ndim-intro/>`_ is also available.

    ---

    **Optimized compute engine**

    It teams with internal compression to compute complex `mathematical expressions <https://ironarray.io/blog/blosc2-eval-expressions>`_ and `reductions <https://www.blosc.org/posts/ndim-reductions/>`_ at high speed.

    ---

    **Support for sparse data**

    For `efficient storage <https://www.blosc.org/docs/Exploring-MilkyWay-SciPy2023.pdf>`_ and `manipulation <https://www.blosc.org/docs/Exploring-MilkyWay-SciPy2023-paper.pdf>`_ of data with many zero values.

    ---

    **Flexible storage**

    Can transparently store data in `memory, disk <https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/01.ndarray-basics.ipynb>`_, or `the network <https://ironarray.io/caterva2>`_. `Memory-mapped files <https://www.blosc.org/python-blosc2/reference/autofiles/storage/blosc2.Storage.html#blosc2.Storage>`_ are also supported.

    ---

    **Two-level partitions**

    Leverages multi-level CPU caches, enhancing `data access <https://www.blosc.org/posts/blosc2-ndim-intro/>`_ and `compute performance <https://www.blosc.org/posts/ndim-reductions/>`_ for modern multi-core processors.

.. panels::
    :card: intro-card text-center no-border
    :column: col-lg-4 col-md-6 col-sm-12 mb-4 offset-lg-4 offset-md-3 d-flex
    :container: + gap-3

    **Uncomplicated format**

    `Blosc2's format <https://github.com/Blosc/c-blosc2/blob/main/README_FORMAT.rst>`_, with specs taking less than 4000 words, makes it easy to read and integrate with other systems and tools.


.. raw:: html

    <h1 class="text-center">Documentation</h1>

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-12 col-sm-12 col-xs-12 d-flex
    :container: + gap-3


    ---

    Getting Started
    ^^^^^^^^^^^^^^^

    New to Python-Blosc2? Check out the getting started guides. They contain an
    introduction to Python-Blosc2 main concepts and different tutorials.

    +++

    .. link-button:: getting_started/index
            :type: ref
            :text: To the getting started guides
            :classes: btn-info

    ---

    API Reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of the Python-Blosc2 API.
    The reference describes how the functions work and which parameters can
    be used.

    +++


    .. link-button:: reference/index
            :type: ref
            :text: To the reference guide
            :classes: btn-info


    ---

    Development
    ^^^^^^^^^^^

    Saw a typo in the documentation? Want to improve
    existing functionalities? The contributing guidelines will guide
    you through the process of improving Python-Blosc2.

    +++

    .. link-button:: development/index
            :type: ref
            :text: To the development guide
            :classes: btn-info

    ---

    Release Notes
    ^^^^^^^^^^^^^

    Want to see what's new in the latest release? Check out the release notes to find out!

    +++

    .. link-button:: release_notes/index
            :type: ref
            :text: To the release notes
            :classes: btn-info



..  toctree::
    :maxdepth: 1
    :hidden:

    Getting Started <getting_started/index>
    API Reference <reference/index>
    Development <development/index>
    Release Notes <release_notes/index>
