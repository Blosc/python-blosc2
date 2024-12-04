.. _title:

.. title:: Python-Blosc2 Documentation

.. raw:: html

    <p style="text-align: center; color: black; background-color: rgba(230, 169, 9, 0.65);">
        <a href="https://github.com/Blosc/python-blosc2/blob/main/RELEASE_NOTES.md"
           style="font-size: 1.5em;">Version 3.0.0 rc2 released on 2024-12-02!</a>
        <span style="display: inline-block; width: 20px;"></span>
        <span style="font-family: monospace;">pip install blosc2==3.0.0rc2</span>
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
    :column: col-lg-4 col-md-6 col-sm-12 mb-4
    :container: + gap-3

    **N-dim & compressed arrays**

    `NDArray objects <https://www.blosc.org/python-blosc2/getting_started/tutorials/01.ndarray-basics.html>`_ allow for efficient storage and manipulation of N-dim data, making it easy to work with large datasets.

    ---

    **Excellent compression**

    Combines `advanced codecs and filters <https://www.blosc.org/posts/bytedelta-enhance-compression-toolset/>`_ for efficient compression, reducing storage space while maintaining high performance.

    ---

    **Optimized compute engine**

    It teams with internal compression to compute complex `mathematical expressions <https://ironarray.io/blog/blosc2-eval-expressions>`_ and `reductions <https://www.blosc.org/posts/ndim-reductions/>`_ at high speed.

    ---

    **Streamlined format**

    `Blosc2's format <https://github.com/Blosc/c-blosc2/blob/main/README_FORMAT.rst>`_, with specs taking less than 4000 words, makes it easy to integrate with other systems and tools.

    ---

    **Flexible storage**

    Can store data in `memory, disk <https://github.com/Blosc/python-blosc2/blob/main/doc/getting_started/tutorials/01.ndarray-basics.ipynb>`_, or `network <https://ironarray.io/caterva2>`_, adapting to your needs and facilitates integration into various systems.

    ---

    **Two-level partitions**

    Leverages multi-level CPU caches, `enhancing data access <https://www.blosc.org/posts/blosc2-ndim-intro/>`_ and `performance <https://www.blosc.org/posts/ndim-reductions/>`_ for modern multi-core processors.


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
