#------------------------------------------------------------------------------
# SLURM script to run the JupyterLab
# The Advanced Research Computing at Hopkins (ARCH)
# Software Team < help@rockfish.jhu.edu >
# Date: Feb, 18 2022
#
# Configuration file for jupyter-notebook.

# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

c.NotebookApp.allow_password_change = True

#c.NotebookApp.keyfile = u'/home/yli21/.jupyter/ssl/arch_rockfish.key'
#c.NotebookApp.certfile = u'/home/yli21/.jupyter/ssl/arch_rockfish.pem'

c.NotebookApp.open_browser = False

# Forces users to use a password for the Notebook server.
c.NotebookApp.password = u'sha1:f8acbc486f37:4d87f1947e7f3fbfc9c123f849c607604ab7c187'

c.NotebookApp.password_required = True
c.NotebookApp.quit_button = True

## The port the notebook server will listen
c.NotebookApp.port_retries = 1

## (sec) Time window used to  check the message and data rate limits.
#c.NotebookApp.rate_limit_window = 3

#  Terminals may also be automatically disabled if the terminado package is not
#  available.
c.NotebookApp.terminals_enabled = True

c.NotebookApp.ip = '*'
c.NotebookApp.port = PORT

