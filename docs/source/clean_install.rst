==================
Clean installation
==================

A number of bash scripts provided in the directory ``dev_scripts`` for cleaning and reinstalling the ``mbircone`` environment.

In order to completely remove ``mbircone``,
``cd`` into ``dev_scripts`` and run the command::

    $ source clean_mbircone.sh

In order to install ``mbircone`` along with requirements for ``mbircone``, demos, and docs,
``cd`` into ``dev_scripts`` and run the command::

    $ source install_mbircone.sh

In order to install documentation that can be viewed from ``mbircone/docs/build/html/index.html``,
``cd`` into ``dev_scripts`` and run the command::

    $ source install_docs.sh

In order to destroy the conda environement named ``mbircone`` and then recreate and activate it,
``cd`` into ``dev_scripts`` and run the command::

    $ source reinstall_conda_environment.sh

In order to destroy and clean everything, and then recreate the conda environment and reinstall ``mbircone`` and its documentation
``cd`` into ``dev_scripts`` and run the command::

    $ source clean_install_all.sh

**Be careful with these last two commands** because they will destroy the conda environment named ``mbircone``.
