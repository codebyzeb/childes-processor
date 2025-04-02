"""Process CHILDES CSV files.

This module provides functionality for processing CHILDES CSV files, including:
- Cleaning and standardizing utterances
- Identifying child and adult utterances
- Transcribing utterances using G2P+
- Creating train/validation splits

The main class ChildesProcessor handles all these operations and can be used to process
either individual CSV files or directories containing multiple CSV files.
"""

import json
from pathlib import Path
import pandas as pd
import re 
import logging

from .dicts import w2string, string2w, punctuation_dict, col2dtype
from g2p_plus import character_split_utterances, transcribe_utterances

DEFAULT_G2P_CONFIG = Path(__file__).parent / 'default_g2p_config.json'

def _clean_english(sentence: str, type: str) -> str:
    """Clean English CHILDES sentences by fixing spelling, punctuation, and word pairs. Based on AOChildes pipeline.py.
    
    Args:
        sentence (str): The utterance to clean
        type (str): The type of utterance (e.g., 'declarative', 'question')
    
    Returns:
        str: The cleaned utterance with standardized spelling and punctuation
    """

    sentence = str(sentence)

    # Fix word pairs 
    for string in string2w:
        if string in sentence:
            sentence = sentence.replace(string, string2w[string])

    # consistent question marking
    if (sentence.startswith('what') and not sentence.startswith('what a ')) or \
            sentence.startswith('where') or \
            sentence.startswith('how') or \
            sentence.startswith('who') or \
            sentence.startswith('when') or \
            sentence.startswith('you wanna') or \
            sentence.startswith('do you') or \
            sentence.startswith('can you'):
        sentence += '?'
    else:
        sentence += f'{punctuation_dict[type]}' if type in punctuation_dict else '.'

    words = []
    for w in str(sentence).split():
        w = w.lower()
        # fix spelling
        if w in w2string:
            w = w2string[w.lower()]
        # split compounds
        w = w.replace('+', ' ').replace('_', ' ')
        words.append(w)
    
    return ' '.join(words)

def _clean(sentence: str, type: str) -> str:
    """Process a non-English CHILDES sentence by lowercasing and adding punctuation.
    
    Args:
        sentence (str): The utterance to clean
        type (str): The type of utterance (e.g., 'declarative', 'question')
    
    Returns:
        str: The cleaned utterance
    """

    sentence = str(sentence)
    if not type in punctuation_dict:
        sentence += '. '
    else:
        sentence += f' {punctuation_dict[type]}'
    words = []
    for w in str(sentence).split():
        w = w.lower()
        # split compounds
        w = w.replace('+', ' ').replace('_', ' ')
        words.append(w)
    return ' '.join(words)

class ChildesProcessor:
    """Processes CHILDES CSV files for phonological analysis.
    
    This class handles the complete pipeline for processing CHILDES data:
    1. Loading and cleaning utterances
    2. Filtering by age and speaker type (child/adult)
    3. Transcribing utterances using G2P+
    4. Creating train/validation splits
    
    Attributes:
        df (pd.DataFrame): The processed CHILDES data
        logger (logging.Logger): Logger for tracking processing steps
    """
    
    def __init__(self, path: Path, keep_child_utterances: bool = True, max_age: int = None):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.df = self.load_data(path)
    
        if max_age is not None:
            num_above_max = len(self.df[self.df['target_child_age'] > max_age])
            self.df = self.df[self.df['target_child_age'] <= max_age]
            self.logger.info(f'Removed {num_above_max} utterances above {max_age} months. Now have {len(self.df)} utterances below this age.')

        if not keep_child_utterances:
            num_child = len(self.df[self.df['is_child']])
            self.df = self.df[~self.df['is_child']]
            self.logger.info(f'Removed {num_child} child utterances. Now have {len(self.df)} adult utterances.')
    
    def load_data(self, path: Path):
        """ Given a path to a CHILDES CSV file, or a folder of CHILDES CSV files, prepare the data for training, returning a DataFrame.
        
        Carries out the following:
        1. Keep only the columns specified in col2dtype.
        2. Remove rows with negative number of tokens or no tokens.
        3. Add a column 'is_child' to indicate whether the speaker is a child.
        4. Sort the DataFrame by target_child_age and transcript_id.
        5. Remove rows that have nonsense words.
        6. Clean each sentence with some simple preprocessing (fixing spelling if in English).

        Args:
            path (Path): Path to a CHILDES CSV file or folder of CHILDES CSV files.
        
        """

        if not path.exists():
            raise FileNotFoundError(f'Path {path} does not exist.')
        if path.is_dir():
            self.logger.info('Path is a directory, will extract utterances from all CSVs found in this directory.')
            transcripts = path
        else:
            self.logger.info('Path is a file, will extract utterances from this CSV.')
            transcripts = [path]

        # Load each utterance as a row in original CSV and remove empty rows
        dfs = [pd.read_csv(csv_path, index_col='id', usecols=col2dtype.keys(), dtype=col2dtype) for csv_path in sorted(transcripts.glob('*.csv'))]
        df = pd.concat(dfs)
        df.drop(df[df['num_tokens'] <= 0].index, inplace=True)
        self.logger.info(f'Loaded {len(df)} utterances from {len(dfs)} CSVs.')
        
        # Add a column to indicate whether the speaker is a child
        roles = df['speaker_role'].unique()
        child_roles = ['Target_Child', 'Child']
        self.logger.info(f'Found speaker roles: {roles}')
        df['is_child'] = df['speaker_role'].isin(child_roles)

        # Sort df by target_child_age and transcript_id
        df.sort_values(by=['target_child_age', 'transcript_id'], inplace=True)

        # Remove rows with ignore_regex in gloss
        ignore_regex = re.compile(r'(�|www|xxx|yyy)')
        df.drop(df[df['gloss'].apply(lambda x: ignore_regex.findall(str(x)) != [])].index, inplace=True)
        
        # Drop null gloss
        df.dropna(subset=['gloss'], inplace=True)

        # Clean each sentence, special cleaning for English
        if df['language'].iloc[0] == 'eng':
            df['processed_gloss'] = df.apply(lambda x: _clean_english(x['gloss'], x['type']), axis=1)
        else:
            df['processed_gloss'] = df.apply(lambda x: _clean(x['gloss'], x['type']), axis=1)

        # Fix some data types
        df['part_of_speech'] = df['part_of_speech'].astype(str)
        df['part_of_speech'] = df['part_of_speech'].apply(lambda x: ' ' if x == 'nan' else x)
        df['stem'] = df['stem'].astype(str)
        df['stem'] = df['stem'].apply(lambda x: ' ' if x == 'nan' else x)
        df['target_child_sex'] = df['target_child_sex'].astype(str)
        df['target_child_sex'] = df['target_child_sex'].apply(lambda x: 'unknown' if x == 'nan' else x)

        # Fix transcription errors in Serbian
        if df['language'].iloc[0] == 'srp':
            df.drop(df[df['processed_gloss'].str.contains('q')].index, inplace=True)

        return df
    
    def transcribe_utterances(self, language: str, keep_word_boundaries: bool = True, verbose: bool = False):
        """ Transcribe utterances. """

        with open(DEFAULT_G2P_CONFIG, 'r') as f:
            default_g2p_config = json.load(f)
        language = language.lower()
        if language not in default_g2p_config:
            raise ValueError(f'Language "{language}" not found in default g2p config. Choices: {list(default_g2p_config.keys())}')
        config = default_g2p_config[language]
        self.logger.info(f'Using g2p-plus config: {config}')

        backend = config['backend']
        lang = config['language']
        lines = self.df['stem'] if lang in ['mandarin', 'cantonese', 'yue-Latn', 'cmn-Latn'] else self.df['processed_gloss']
        if 'wrapper_kwargs' in config:
            self.df['ipa_transcription'] = transcribe_utterances(lines, backend, lang, keep_word_boundaries=keep_word_boundaries, verbose=verbose, **config['wrapper_kwargs'])
        else:
            self.df['ipa_transcription'] = transcribe_utterances(lines, backend, lang, keep_word_boundaries=keep_word_boundaries, verbose=verbose)

        num_empty = len(self.df[self.df['ipa_transcription'] == ''])
        num_empty += len(self.df[self.df['ipa_transcription'] == 'WORD_BOUNDARY '])
        if num_empty > 0:
            self.logger.warning(f'{num_empty} lines were not transcribed successfully. Dropping these.')
            self.df = self.df[self.df['ipa_transcription'] != '']
            self.df = self.df[self.df['ipa_transcription'] != 'WORD_BOUNDARY ']

    def character_split_utterances(self):
        """ Character split utterances. """

        self.df['character_split_utterance'] = character_split_utterances(self.df['processed_gloss'])

    def split_df(self, dev_size: int = 10_000, sequential: bool = False):
        """ Split the DataFrame into a training set and a validation set.
        
        Note that the DataFrame is likely to be sorted by age, so the split will be age-ordered
        and if the split is sequential, the validation set will consist of utterances
        targetted at older children.
        """

        if sequential:
            train = self.df[:-dev_size]
            valid = self.df[-dev_size:]
        else:
            interval = len(self.df) // dev_size
            self.logger.info("Taking every {}th line to get 10,000 lines for validation...".format(interval))
            valid = self.df.iloc[::interval]
            valid = valid[:dev_size]
            train = self.df.drop(valid.index)
        return train, valid

    def print_statistics(self):
        """ Print statistics about the DataFrame. """

        total_corpora = len(self.df['corpus_id'].unique())
        total_speakers = len(self.df['speaker_id'].unique())
        total_target_children = len(self.df['target_child_id'].unique())
        total_lines = len(self.df)
        num_words = sum([line.count('WORD_BOUNDARY') for line in self.df['ipa_transcription']])
        num_phonemes = sum([len(line.split()) for line in self.df['ipa_transcription']]) - num_words

        self.logger.info(f'Total corpora: {total_corpora}')
        self.logger.info(f'Total speakers: {total_speakers}')
        self.logger.info(f'Total target children: {total_target_children}')
        self.logger.info(f'Total lines: {total_lines}')
        self.logger.info(f'Total words: {num_words}')
        self.logger.info(f'Total phonemes: {num_phonemes}')

    def save_df(self, out_path: Path):
        """ Save the DataFrame to a CSV file. """

        out_path.mkdir(exist_ok=True, parents=True)
        self.df.to_csv(out_path / 'processed.csv', index=False)
        self.logger.info(f'Saved processed dataset to {out_path / "processed.csv"} with a total of {len(self.df)} utterances.')

    def save_splits(self, out_path: Path):
        """ Save the training and validation DataFrames to CSV files. """

        out_path.mkdir(exist_ok=True, parents=True)
        train, valid = self.split_df()
        train.to_csv(out_path / 'train.csv')
        valid.to_csv(out_path / 'valid.csv')
        self.logger.info(f'Saved train and valid sets to {out_path}')