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

# Add version banner for development versions
if 'dev' in release or '+' in release:
    html_theme_options = {
        "announcement": f"<em>Development version ({release})</em> - For stable release, see tagged versions.",
    }

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  
    'sphinx.ext.napoleon',  
    'sphinx.ext.viewcode',   
    'sphinx.ext.mathjax',     
    'sphinx_autodoc_typehints', 
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_css_files = [
    "version_tabs.css",
]


DOCS_CHANNEL = os.environ.get("DOCS_CHANNEL", "latest")
DOCS_ROOT = os.environ.get("DOCS_ROOT", "").rstrip("/")

if DOCS_ROOT:
    stable_url = f"{DOCS_ROOT}/stable/"
    latest_url = f"{DOCS_ROOT}/latest/"
else:
    stable_url = "../stable/"
    latest_url = "../latest/"

html_theme_options = globals().get("html_theme_options", {})


def _version_tabs_html(channel: str) -> str:
    def cls(name: str) -> str:
        return "version-tab selected" if channel == name else "version-tab"

    return f"""
    <div class="version-tabs-container">
      <a href="{stable_url}" class="{cls('stable')}">Stable</a>
      <a href="{latest_url}" class="{cls('latest')}">Latest (dev)</a>
    </div>
    """

_base_announcement = html_theme_options.get("announcement", "")
html_theme_options["announcement"] = _version_tabs_html(DOCS_CHANNEL) + _base_announcement


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
