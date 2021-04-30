python-blosc2 release procedure
===============================

Preliminaries
-------------
* Make sure that the c-blosc2 submodule is updated to the latest version (or a specific version that
will be documented in the `RELEASE_NOTES.md`)::

    cd blosc2/c-blosc2
    git switch <desired branch or tag>
    cd -
    git commit -m "Update c-blosc2 sources" blosc2/c-blosc2
    git push

* Make sure that the current master branch is passing the tests in continuous integration.

* Make sure that `RELEASE_NOTES.md` and `ANNOUNCE.rst` are up to date with the latest news
  in the release.

* Check that `VERSION` file contains the correct number.

* Check any copyright listings and update them if necessary. You can use ``grep
  -i copyright`` to figure out where they might be.

* Commit the changes::

    git commit -a -m "Getting ready for release X.Y.Z"
    git push

* Check that the documentation is correctly created in https://python-blosc2.rtfd.io.


Tagging
-------

* Create a signed tag ``X.Y.Z`` from ``master``.  Use the next message::

    git tag -a vX.Y.Z -m "Tagging version X.Y.Z"

* Push the tag to the github repo::

    git push
    git push --tags

After the tag would be up, update the release notes in: https://github.com/Blosc/python-blosc2/releases

Packaging
---------

* Make sure that you are in a clean directory.  The best way is to
  re-clone and re-build::

    cd /tmp
    git clone --recursive https://github.com/Blosc/python-blosc2.git
    cd python-blosc2
    python setup.py build_ext

* Check that all Cython generated ``*.c`` files are present.

* Make the tarball with the command::

    python setup.py sdist
    pip install dist/blosc2-X.Y.Z.tar.gz

Do a quick check that the tarball is sane.


Uploading
---------

* Register and upload it also in the PyPi repository::

    twine upload dist/*


It takes about 15 minutes for it to be installed using::

    pip install blosc2



Announcing
----------

* Send an announcement to the Blosc list.  Use the ``ANNOUNCE.rst`` file as skeleton
  (or possibly as the definitive version).

* Announce in Twitter via @Blosc2 account and rejoice.


Post-release actions
--------------------

* Change back to the actual python-blosc2 repo::

    cd $HOME/blosc/python-blosc2


* Create new headers for adding new features in ``RELEASE_NOTES.md``
  add this place-holder:

  XXX version-specific blurb XXX

* Edit ``VERSION`` in master to increment the version to the next
  minor one (i.e. X.Y.Z --> X.Y.(Z+1).dev0).

* Commit your changes with::

    git commit -a -m "Post X.Y.Z release actions done"
    git push


That's all folks!