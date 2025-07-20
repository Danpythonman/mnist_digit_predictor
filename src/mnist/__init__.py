'''
Top-level package for the MNIST digit prediction project.

If you want to ignore ModuleNotFound errors when importing modules related to
the model (for example, if you are only running inference on the FastAPI server
without PyTorch), then make sure you set the IGNORE_INITIAL_MODULENOTFOUND
before running the server.
'''

# !!! IMPORTANT !!!
#
# We are importing in this way to allow the FastAPI server to start up
# without having the Python dependencies that are used when training the
# models.
#
# For example, we use PyTorch when training, but for inference we don't need
# it. But when we start up the server using Uvicorn it will start executing
# this file as the parent package, attempt to import torch, and subsequently
# crash because it's not installed. Thus, we want to be able to ignore these
# errors when running the FastAPI server.
#
# However, we don't want to *always* ignore these errors, because they are
# valid when training. Hence, we wrap the imports in a try/except block and
# set an environment variable that signals when we can ignore the import
# errors.
#
# So if you want the server to start up without having to install PyTorch
# and other training dependencies, make sure you set the
# IGNORE_INITIAL_MODULENOTFOUND environment variable when you run the
# server.
try:
    from . import model
    from . import utils
except ModuleNotFoundError as e:
    import os
    if os.getenv('IGNORE_INITIAL_MODULENOTFOUND', False):
        pass
    else:
        raise e
