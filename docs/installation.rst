.. highlight:: shell

============
Installation
============


Stable release
----------------

**Option 1 - recommended for users**

To install cellmaps_vnn, run this command in your terminal:

.. code-block:: console

    $ pip install cellmaps_vnn

**This is the preferred method to install cellmaps_vnn**, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
-------------

**Option 2.1 - recommended for developers**

The sources for cellmaps_vnn can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/idekerlab/cellmaps_vnn.git

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/idekerlab/cellmaps_vnn/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


Using Makefile
---------------

**Option 2.2 - recommended for developers**

The package provides a Makefile that could be used for installation and building of cellmaps_vnn

.. code-block::

    git clone https://github.com/idekerlab/cellmaps_vnn.git
    cd cellmaps_vnn
    pip install -r requirements_dev.txt
    make clean dist
    pip install dist/cellmaps_pipeline*whl

Each time you make code chages you need to unintall and install the package again.

.. code-block::

    pip uninstall cellmaps_vnn -y; make clean dist; pip install dist/cellmaps_pipeline*whl

.. _Github repo: https://github.com/idekerlab/cellmaps_vnn
.. _tarball: https://github.com/idekerlab/cellmaps_vnn/tarball/master
