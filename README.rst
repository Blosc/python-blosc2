=============
Python-Blosc2
=============

A fast & compressed ndarray library with a flexible compute engine
==================================================================

:Author: The Blosc development team
:Contact: blosc@blosc.org
:Github: https://github.com/Blosc/python-blosc2
:Actions: |actions|
:PyPi: |version|
:NumFOCUS: |numfocus|
:Code of Conduct: |Contributor Covenant|

.. |version| image:: https://img.shields.io/pypi/v/blosc2.svg
        :target: https://pypi.python.org/pypi/blosc2
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
        :target: https://github.com/Blosc/community/blob/master/code_of_conduct.md
.. |numfocus| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
        :target: https://numfocus.org
.. |actions| image:: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml/badge.svg
        :target: https://github.com/Blosc/python-blosc2/actions/workflows/build.yml


What is Python-Blosc2?
=======================

Python-Blosc2 is a high-performance compressed ndarray library with a flexible
compute engine, using `C-Blosc2 <https://www.blosc.org/c-blosc2/c-blosc2.html>`_
as its compression backend. It allows complex calculations on compressed data,
whether stored in memory, on disk, or over the network (e.g., via
`Caterva2 <https://github.com/ironArray/Caterva2>`_).  It uses the
`C-Blosc2 simple and open format
<https://github.com/Blosc/c-blosc2/blob/main/README_FORMAT.rst>`_ for storing
compressed data.

More info: https://www.blosc.org/python-blosc2/getting_started/overview.html

Installing
==========

Binary packages are available for major OSes (Win, Mac, Linux) and platforms.
Install from PyPi using ``pip``:

.. code-block:: console

    pip install blosc2 --upgrade

Conda users can install from conda-forge:

.. code-block:: console

    conda install -c conda-forge python-blosc2

Documentation
=============

The documentation is available here:

https://blosc.org/python-blosc2/python-blosc2.html

You can find examples at:

https://github.com/Blosc/python-blosc2/tree/main/examples

A tutorial from PyData Global 2024 is available at:

https://github.com/Blosc/Python-Blosc2-3.0-tutorial

It contains Jupyter notebooks explaining the main features of Python-Blosc2.

License
=======

This software is licensed under a 3-Clause BSD license. A copy of the
python-blosc2 license can be found in
`LICENSE.txt <https://github.com/Blosc/python-blosc2/tree/main/LICENSE.txt>`_.

Discussion forum
================

Discussion about this package is welcome at:

https://github.com/Blosc/python-blosc2/discussions

Social feeds
------------

Stay informed about the latest developments by following us in
`Mastodon <https://fosstodon.org/@Blosc2>`_,
`Bluesky <https://bsky.app/profile/blosc.org>`_ or
`LinkedIn <https://www.linkedin.com/company/88381936/admin/dashboard/>`_.

Thanks
======

Blosc2 is supported by the `NumFOCUS foundation <https://numfocus.org>`_, the
`LEAPS-INNOV project <https://www.leaps-innov.eu>`_
and `ironArray SLU <https://ironarray.io>`_, among many other donors.
This allowed the following people have contributed in an important way
to the core development of the Blosc2 library:

- Francesc Alted
- Marta Iborra
- Aleix Alcacer
- Oscar Guiñón
- Juan David Ibáñez
- Ivan Vilata i Balaguer
- Oumaima Ech.Chdig

In addition, other people have participated to the project in different
aspects:

- Jan Sellner, contributed the mmap support for NDArray/SChunk objects.
- Dimitri Papadopoulos, contributed a large bunch of improvements to the
  in many aspects of the project.  His attention to detail is remarkable.
- And many others that have contributed with bug reports, suggestions and
  improvements.

Citing Blosc
============

You can cite our work on the various libraries under the Blosc umbrella as follows:

.. code-block:: console

  @ONLINE{blosc,
    author = {{Blosc Development Team}},
    title = "{A fast, compressed and persistent data store library}",
    year = {2009-2025},
    note = {https://blosc.org}
  }

Donate
======

If you find Blosc useful and want to support its development, please consider
making a donation via the `NumFOCUS <https://numfocus.org/donate-to-blosc>`_
organization, which is a non-profit that supports many open-source projects.
Thank you!


**Compress Better, Compute Bigger**
