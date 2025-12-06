# docs/source/conf.py

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Econox'
copyright = '2025, Haruto Ito'
author = 'Haruto Ito'

init_path = os.path.join(os.path.dirname(__file__), '../../src/econox/__init__.py')

# Extract version information from __init__.py
release = ''
with open(init_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            release: str = line.split('=')[1].strip().strip('"').strip("'")
            break
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  
    'sphinx.ext.napoleon',  
    'sphinx.ext.viewcode',   
    'sphinx.ext.mathjax',     
    'sphinx_autodoc_typehints', 
]

# Mock imports for autodoc
autodoc_mock_imports = [
    "jax",
    "jax.numpy",
    "jax.scipy",
    "jax.flatten_util",
    "jax.experimental",
    "equinox",
    "jaxtyping",
    "numpy",
    "pandas",
    "optax",
    "optimistix",
    "lineax",
    "scipy",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True 

autodoc_typehints = 'description'  
