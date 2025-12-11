import sys
import re
import os
from datetime import date

def update_file(filepath, pattern, replacement):
    """
    Reads a file, replaces text matching the pattern, and writes it back if changed.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Perform regex replacement
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != new_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes needed for {filepath}")

def main():
    # Ensure a version argument is provided
    if len(sys.argv) < 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    print(f"Updating files to version: {version}")
    

    # 1. Update CITATION.cff
    # Update the 'version' field
    update_file("CITATION.cff", r"^version: .*", f"version: {version}")
    
    # Update the 'date-released' field to today's date
    today = date.today().isoformat()
    current_year = date.today().year
    update_file("CITATION.cff", r"^date-released: .*", f"date-released: {today}")

    # 2. Update README.md
    # Update the version inside the BibTeX citation block: version = {x.x.x}
    update_file("README.md", r"version = \{.*?\}", f"version = {{{version}}}")
    update_file("README.md", r"year = \{\d{4}\}", f"year = {{{current_year}}}")

if __name__ == "__main__":
    main()