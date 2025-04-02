# CHILDES Processor

Scripts for processing the CHILDES dataset and converting it to a phonemic representation. Used to create the [IPA-CHILDES](https://huggingface.co/datasets/phonemetransformers/ipa-childes) dataset (see `scripts/create_ipa_childes`).

## Installation

The simplest way is using pip:

```
pip install childes-processor
```

Or you can install from source:

```
git clone https://github.com/codebyzeb/childes-processor
cd childes-processor
pip install .
```

### Dependencies

If using the `process` command to convert CHILDES to IPA, you may require additional dependencies for [G2P+](https://github.com/codebyzeb/g2p-plus).

If you are using the `download`, make sure you have R installed.

# Usage

CHILDES processor can be used as a command-line interface using `childes-processor` or by importing `ChildesDownloader` or `ChildesProcessor` in python. The CLI has three modes: *download*, *process* and *extract*, allowing the user to download and transcribe the CHILDES dataset. 

To bring up the help menu, simply type:

```
childes_processor -h
```

Or for each mode, there is also a help menu:

```
childes_processor extract -h
```

## Download

The **download** mode allows for corpora to be downloaded from CHILDES. For example, to download the _Warren_ corpus from the _Eng-NA_ collection, run the following:

```
childes_processor download Eng-NA --corpus Warren -o childes/downloaded
```

This will save the utterances to `downloaded/Eng-NA/Warren.csv`. If `-s` is used, the data will be separated by speaker. The command can also be run without the corpus provided, downloading all corpora available in the collection:

```
childes_processor download Eng-NA -o downloaded
```

## Process

The *process* mode will process downloaded CSVs from CHILDES (those downloaded from the **download** tool) and provide a new CSV with additional columns and utterances sorted by child age. The additional columns are as follows:

| Column | Description |
|:----|:-----|
| `is_child`| Whether the utterance was spoken by a child or not. Note that unless the `-k` or `--keep` flag is set, all child utterances will be dicarded so this column will only contain `False`. |
| `processed_gloss`| The pre-processed orthographic utterance. This includes lowercasing, fixing English spelling and adding punctuation marks. This is based on the [AOChildes](https://github.com/UIUCLearningLanguageLab/AOCHILDES) preprocessing.|
| `ipa_transcription`| A phonemic transcription of the utterance in IPA, space-separated with word boundaries marked with the `WORD_BOUNDARY` token. This uses [G2P+](https://github.com/codebyzeb/g2p-plus) using specifically-configured backends and language codes. |
| `character_split_utterance`| A space separated transcription of the utterance, produced simply by splitting the processed gloss by character. This is intended to have a very similar format to `ipa_transcription` for studies comparing phonemic to orthographic transcriptions. |

The first required argument is the CSV or folder of CSVs to process. The second argument is the language that will be used for producing the phonemic transcription. To view supported languages, use `-h`. 

The `-k` or `--keep` flag is used to keep child utterances. The `-s` or `--split` flag is used to split the resulting dataset into training set and a validation set containing 10,000 utterances. The `-m` or `--max_age` flag is used to discard all utterances produced when the child's age greater than the provided number of months.

For example, to process all downloaded _Eng-NA_ corpora, run the following:

```
childes_processor process downloaded/Eng-NA EnglishNA -o processed/Eng-NA -s
```

This will take all the CSVs in the `downloaded/Eng-NA` folder and create two new CSVs, `train.csv` and `valid.csv` in the `processed/Eng-NA` folder specified containing processed utterances and additional useful information. These datasets contain phonemic transcriptions of each utterance that have been produced using the `en-us` language backend. If the path provided is a CSV instead of a folder, just that CSV will be processed.

## Extract

The **extract** mode will take a CSV dataset and produce a text file containing a column from that CSV dataset. It has the option use a maximum cutoff, as with the process mode, using `-m` or `--max_age`. The intended use is to gather all phonemic or orthographic utterances from the processed dataset (but can also be used to extract other columns, or to extract from a downloaded CSV that hasn't been processed). 

For example, to extract all ipa transcriptions from the train file produced by the previous example, only including utterances targeting children under the age of 2, run the following:

```
childes_processor extract processed/Eng-NA/train.csv ipa_transcription -o extracted/Eng-NA -m 24
```

This will create a file `childes/extracted/Eng-NA/utterances.txt` containing the contents of the `ipa_transcription` column where `target_child_age` is less than 24 months.

## Python Usage

The *download* and *process* modes can also be used within Python. For example:

```python
from childes_processor import ChildesProcessor, ChildesDownloader
from pathlib import Path
DOWNLOAD_PATH = Path('downloaded')
PROCESSED_PATH = Path('processed')

downloader = ChildesDownloader()
downloader.download('Eng-NA',
                    'Warren',
                    DOWNLOAD_PATH,
                    separate_by_child=False)

processor = ChildesProcessor(DOWNLOAD_PATH / 'Eng-NA',
                             keep_child_utterances=True,
                             max_age = 120)
processor.transcribe_utterances('EnglishNA')
processor.character_split_utterances()
processor.print_statistics()
processor.save_df(PROCESSED_PATH / 'Eng-NA')

```