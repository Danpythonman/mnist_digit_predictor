MNIST Digit Prediction Notebooks
================================

These are the Jupyter notebooks used to learn, explore, and analyze to make the best neural network for digit prediction.

Currently there are two notebooks:

- [exploration.ipynb](./exploration.ipynb) - initial exploration on using neural networks to predict handwritten digits

- [improving.ipynb](./improving.ipynb) - improving the model previously used for predicting digits.

Common Commands Used for Jupyter Notebooks
------------------------------------------

There are a few commands I use often when working with Jupyter notebooks. I'm including them here as a form of documentation.

### Stripping Outputs

I like to strip out the outputs from Jupyter notebooks before committing them to Git. The command is:

```bash
nbstripout path/to/notebook.ipynb
```

You may need to install `nbstripout` first:

```bash
pip install --upgrade nbstripout
```

### Converting to PDF

To convert a Jupyter notebook (that was already run and has outputs in it) to a PDF:

```bash
jupyter nbconvert --to pdf path/to/notebook.ipynb
```

### Converting to HTML

To convert a Jupyter notebook (that was already run and has outputs in it) to an HTML page:

```bash
jupyter nbconvert --to html --template classic path/to/notebook.ipynb
```

Since the HTML pages are committed to Git and published on the main website, the page can be compressed with NPM's `html-minifier-terser` command:

```bash
html-minifier-terser path/to/notebook.html -o path/to/notebook.min.html --collapse-whitespace --remove-comments --minify-css --minify-js
```

You may need to install `html-minifier-terser` first:

```bash
npm install -g html-minifier-terser
```
