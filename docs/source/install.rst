############
Installation
############

Before installing ``CAFE``, we recommend creating a conda environment and then install the last tagged version of ``CAFE`` from the GitHub repository master branch using the following commands:

.. code-block:: bash

   conda create -n <cafe_env> python=3.11.5 scipy=1.11.4
   conda activate <cafe_env>
   pip install git+https://github.com/GOALS-survey/CAFE.git

Note that the last command above will NOT update ``CAFE`` to the latest version posted in GitHub if you already have ``CAFE`` installed in your environment from a previous version. If you really want to install the latest developer version available at the time, you can execute this command to force ``pip`` do it:

.. code-block:: bash

   pip install --upgrade --force-reinstall git+https://github.com/GOALS-survey/CAFE.git

However, we warn that **the developer version of ``CAFE`` is not supported**. We only provide support for the latest stable version (currently v1.0.0).

Also, you may want to install jupyter lab to run the tutorial notebook:

.. code-block:: bash
   
   pip install jupyterlab
