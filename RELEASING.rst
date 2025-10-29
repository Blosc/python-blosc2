Python-Blosc2 release procedure
===============================

Preliminaries
-------------

* Set the version number for the release by using::

    python update_version.py X.Y.Z

  and double-check the updated version number in ``pyproject.toml`` and with::

    python -c "import blosc2; print(blosc2.__version__)"

* Make sure that the c-blosc2 repository is updated to the latest version (or a specific
  version that will be documented in the ``RELEASE_NOTES.md``). In ``CMakeLists.txt`` edit::

    FetchContent_Declare(blosc2
        GIT_REPOSITORY https://github.com/Blosc/c-blosc2
        GIT_TAG b179abf1132dfa5a263b2ebceb6ef7a3c2890c64
    )

  to point to the desired commit/tag in the c-blosc2 repo.

* Make sure that the current main branch is passing the tests in continuous integration.

* Build the package and make sure that tests are passing::

    pip install -e ".[test]"
    pytest

* Make sure that ``RELEASE_NOTES.md`` and ``ANNOUNCE.rst`` are up to date with the
  latest news in the release.

* Commit the changes::

    git commit -a -m "Getting ready for release X.Y.Z"
    git push

* Double check that the supported Python versions for the wheels are the correct ones
  (``.github/workflows/cibuildwheels.yml``).  Add/remove Python version if needed.
  Also, update the ``classifiers`` field in pyproject.toml for the supported Python
  versions.

* Check that the metainfo for the package is correct::

    pipx run build --sdist
    twine check --strict dist/*


Tagging
-------

* Create a (signed, if possible) tag ``X.Y.Z`` from ``main``.  Use the next message::

    git tag -a vX.Y.Z -m "Tagging python-blosc2 version X.Y.Z"

* Push the tag to the github repo::

    git push --tags

* If you happen to have to delete the tag, such as artifacts demonstrates a fault, first delete it locally::

    git tag --delete vX.Y.Z

  and then remotely on Github:

    git push --delete origin vX.Y.Z

* Make sure that the tag is passing the tests in continuous integration (this
  may take about 30 min).

* In case the automatic upload to PyPI fails, you can upload the package
  wheels (and tarball!) by downloading the artifacts manually, copying to
  an empty dir (say dist), and upload to PyPI with::

    rm wheelhouse/*
    # download artifacts from the tag in github
    twine upload --repository blosc2 wheelhouse/*

* Update the latest release in the ``doc/python-blosc2.rst`` file with the new version
  number and date.  Do a commit::

    git commit -a -m "Update latest release in doc"
    git push

* Go to ``https://github.com/Blosc/blogsite`` repo, then to "Actions", click
  on the most recent workflow run (at the top of the list), and then click on
  the "Re-run all jobs" button to regenerate the documentation and check that
  it has been correctly updated in https://www.blosc.org.


Checking packaging
------------------

* Check that the package (and wheels!) have been uploaded to PyPI
  (they should have been created when GHA would finish the tag trigger):
  https://pypi.org/project/blosc2/

* Check that the packages and wheels are sane::

    pip install blosc2[test] -U
    python -c "import blosc2; blosc2.print_versions()"
    pytest

* Do an actual release in github by visiting:
  https://github.com/Blosc/python-blosc2/releases/new
  Add the notes specific for this release.

  Also, upload the wasm32 wheels to release page in github::

    [upload e.g.:] blosc2-3.2.0-cp312-cp312-pyodide_2024_0_wasm32.whl

The wheels may be downloaded by going to "Actions->Python wheels for WASM"
and selecting the completed workflow run for the version you are releasing.
Then, go to the "Artifacts" dropdown and download the WASM wheel file(s).
**Note**: be sure to upload the wheel files, not the zip file containing them.

Announcing
----------

* Send an announcement to the Blosc and PyData lists.  Use the ``ANNOUNCE.rst`` file as
  skeleton (or possibly as the definitive version). Start the subject with ANN:.

* Announce in Mastodon via https://fosstodon.org/@Blosc2 account and rejoice.
  Announce it in Bluesky too.


Post-release actions
--------------------

* Change back to the actual python-blosc2 repo::

    cd $HOME/blosc/python-blosc2

* Create a new header for adding new features in ``RELEASE_NOTES.md``
  with a placeholder text::

    ## Changes from X.Y.Z to X.Y.(Z+1)

    XXX version-specific blurb XXX

* Update the version number in ``pyproject.toml`` and ``version.py`` to the next version number::

    python update_version.py X.Y.(Z+1).dev0

* Commit your changes with::

    git commit -a -m "Post X.Y.Z release actions done"
    git push


Other packaging
---------------

* If you want to package the Python-Blosc2 for conda, you should get an automatic
  message from the conda-forge bot, which will create a pull request.  For releases
  that do not update the C-blosc2 version, you can just merge the pull request;
  otherwise, it is best to wait until the new C-blosc2 version makes its way to
  conda-forge.

* If you want to package Blosc2 for Pyodide, you can use the repo at:
  https://github.com/Blosc/pyodide-recipes
  and update the recipe for the new version.  Then, issue a pull request to upstream.


That's all folks!
