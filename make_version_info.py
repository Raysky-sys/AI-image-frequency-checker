"""
make_version_info.py
====================

Generates `version_info.txt`, the Windows version resource that PyInstaller
embeds into the executable so that File > Properties shows the correct
product name, version, and author.

Run directly:
    python make_version_info.py

The file is written next to this script. It is a build artefact and should
not be committed to the repository.

Author: Raysky-sys
"""

import os
import sys

# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------
VERSION = (1, 9, 2, 0)
VERSION_STRING = ".".join(str(part) for part in VERSION)

COMPANY = "Raysky-sys"
FILE_DESCRIPTION = "AI Image Frequency Checker"
INTERNAL_NAME = "AIImageFrequencyChecker"
ORIGINAL_FILENAME = "AIImageFrequencyChecker.exe"
PRODUCT_NAME = "AI Image Frequency Checker"

# ---------------------------------------------------------------------------
# Template
#
# This is the format PyInstaller's versioninfo parser expects: a Python
# literal that it reads with ast.parse(). Generating it from Python avoids
# all of the parenthesis and quoting problems that occur when the same
# text is echoed from a batch script.
# ---------------------------------------------------------------------------
TEMPLATE = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version},
    prodvers={version},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [StringStruct('CompanyName', '{company}'),
           StringStruct('FileDescription', '{description}'),
           StringStruct('FileVersion', '{version_string}'),
           StringStruct('InternalName', '{internal_name}'),
           StringStruct('OriginalFilename', '{original_filename}'),
           StringStruct('ProductName', '{product}'),
           StringStruct('ProductVersion', '{version_string}')])
      ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def main() -> int:
    """Write version_info.txt. Returns a process exit code."""
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "version_info.txt")

    content = TEMPLATE.format(
        version=VERSION,
        version_string=VERSION_STRING,
        company=COMPANY,
        description=FILE_DESCRIPTION,
        internal_name=INTERNAL_NAME,
        original_filename=ORIGINAL_FILENAME,
        product=PRODUCT_NAME,
    )

    try:
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError as error:
        print(f"          [ERROR] Could not write {target}")
        print(f"          {error}")
        return 1

    print(f"          Written: {target}")
    print(f"          Version: {VERSION_STRING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
