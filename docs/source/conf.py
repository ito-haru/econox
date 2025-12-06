# docs/source/conf.py

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import importlib.metadata
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Econox'
copyright = '2025, Haruto Ito'
author = 'Haruto Ito'

init_path = os.path.join(os.path.dirname(__file__), '../../src/econox/__init__.py')

# Get version from package metadata
try:
    release = importlib.metadata.version('econox')
except importlib.metadata.PackageNotFoundError:
    release = '0.0.0'

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
napoleon_use_ivar = True

autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_preserve_defaults = True
autodoc_member_order = 'bysource'


# Custom processing of signatures to better display equinox.Module fields
def process_signature(app, what, name, obj, options, signature, return_annotation):
    # Only process classes
    if what != "class":
        return signature, return_annotation

    try:
        # Check if it is an Equinox class
        # (For safety, get the module name as a string and check)
        if hasattr(obj, "__module__") and "equinox" in getattr(obj, "__module__", ""):
            import dataclasses
            
            if dataclasses.is_dataclass(obj):
                fields = dataclasses.fields(obj)
                sig_parts = []
                for f in fields:
                    if f.default is not dataclasses.MISSING:
                        sig_parts.append(f"{f.name}={f.default}")
                    elif f.default_factory is not dataclasses.MISSING:
                        sig_parts.append(f"{f.name}=<factory>")
                    else:
                        sig_parts.append(f"{f.name}")
                
                return f"({', '.join(sig_parts)})", return_annotation
    except Exception:
        pass
            
    return signature, return_annotation

# Register the event handler
def setup(app):
    app.connect("autodoc-process-signature", process_signature)
