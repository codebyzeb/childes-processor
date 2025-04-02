"""CHILDES Processor

A tool for processing CHILDES data, including:
- Downloading corpora from CHILDES database
- Cleaning and standardizing utterances
- Converting utterances to IPA using G2P+
- Creating train/validation splits for language modeling

The main classes are:
- ChildesDownloader: For downloading CHILDES corpora
- ChildesProcessor: For processing downloaded corpora
"""

__version__ = "0.1.0"
__author__ = 'Zebulon Goriely'

from .src.processor import ChildesProcessor
from .src.downloader import ChildesDownloader