MNIST Digit Prediction
======================

This project is a handwritten digit prediction system using the MNIST dataset for training.

Currently, a convolutional neural network is being used in the production website [mnist.danieldigiovanni.com/](https://mnist.danieldigiovanni.com/). The network achieves over 99% accuracy on the MNIST validation set.

Main Parts of this Repository
-----------------------------

The main parts of this repository are:

- [./src/](./src/) - The main Python source code for the neural networks, data loading, training loops, loss evaluation, plotting utility functions, and a Flask wrapper.

- [./frontend/](./frontend/) - The frontend for this project. Base HTML, CSS, and JavaScript. Using Vite as the build tool. For more information, see the [frontend README](./frontend/README.md).

- [./notebooks/](./notebooks/) - The Jupyter notebooks used to learn and explore the usage of neural networks for predicting handwritten digits. For more information, see the [notebooks README](./notebooks/README.md).

- [./aws](./aws/) - Scripts and directions on deploying the backend to AWS Lambda. For more information, see the [AWS README](./aws/README.md).

Running the Notebooks
---------------------

To run the Jupyter notebooks in this project, I recommend using Conda to install the dependencies. From there you can use your favorite Jupyter notebook editor.

To create a Conda environment  with all the dependencies I used:

```bash
conda env create -f environment-full.yml
```

Note that [environment-full.yml](./environment-full.yml) contains all the dependencies, including dependencies installed from Pip, and therefore reflects the most accurate set of dependencies I used when working with the notebooks. However, this specification may contain unnecessary dependencies that I no longer use and forgot to uninstall. The smaller [environment.yml](environment.yml) includes only the dependencies I explicitly installed via the `conda install` command.

If using Pip, you can install from [requirements.txt](./requirements.txt):

```bash
pip install -r requirements.txt
```

Similar to the large Conda environment file, the requirements file may contain extra dependencies that are no longer used.

Running the Flask Server
------------------------

The Flask server needs a different (and smaller) set of dependencies than the dependencies required to run the notebooks. These dependencies are listed in [pyproject.toml](./pyproject.toml).

To install the dependencies for the Flask server, all you need to run is:

```bash
pip install -e .
```
To run the server, you can use the following command:

```bash
IGNORE_INITIAL_MODULENOTFOUND=true \
    MODEL_PATH=path/to/model.onnx \
    uvicorn src.mnist.api.routes:app --host 0.0.0.0 --port 8002
```

> For an explanation as to why the `IGNORE_INITIAL_MODULENOTFOUND=true` environment variable is used read the comments in [src/mnist/__init__.py](./src/mnist/__init__.py).

From here, you can send requests to the backend, either through Curl or the frontend.

Contributing
------------

Contributions are welcome! Feel free to open issues or submit pull requests.

License
-------

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Contact
-------

If you have any questions or feedback, feel free to connect with me on LinkedIn at [linkedin.com/in/daniel-di-giovanni/](https://www.linkedin.com/in/daniel-di-giovanni/) or send me an email at [dannyjdigio@gmail.com](mailto:dannyjdigio@gmail.com).
