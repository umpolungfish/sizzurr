from setuptools import setup, find_packages

setup(
    name="sizzurr",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "beautifulsoup4",
        "html2text",
        "markdownify",
        "PyMuPDF",
        "pdfplumber",
        "torch",
        "transformers",
        "Pillow",
        "opencv-python",
        "pyyaml",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "sizzurr = sizzurr.cli:main",
        ],
    },
)
