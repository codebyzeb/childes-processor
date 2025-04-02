import logging

class ChildesDownloader:
    """Downloads utterances from CHILDES using the childespy package.
    
    This class provides functionality to download utterances from the CHILDES database,
    either for an entire collection or a specific corpus within a collection. The downloaded
    utterances can be saved either as a single CSV file or as separate CSV files for each child.
    
    Note:
        The childespy package is imported only when the class is instantiated since it 
        re-downloads childesr each time it's imported.
    """

    def __init__(self):
        """Initialize the ChildesDownloader with logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info('Importing childespy')
        from childespy import get_utterances
        self.get_utterances = get_utterances

    def download(self, collection: str, corpus: str, out_path: str, separate_by_child: bool):
        """Download utterances from CHILDES and save them to CSV file(s).

        Args:
            collection (str): Name of the CHILDES collection (e.g., "Eng-NA", "Romance")
            corpus (str, optional): Name of specific corpus within the collection. If None,
                downloads all corpora in the collection.
            out_path (str): Directory path where the CSV file(s) will be saved
            separate_by_child (bool): If True, creates separate CSV files for each child.
                If False, creates a single CSV file for all utterances.

        The output files will be organized as follows:
        - If separate_by_child=False: {out_path}/{collection}/{corpus or collection}.csv
        - If separate_by_child=True: {out_path}/{collection}/{corpus}/{child_name}.csv
        """
        self.logger.info(f'\n\nAttempting to get utterances from the "{corpus}" corpus in the "{collection}" collection:\n')
        utts = self.get_utterances(collection=collection, corpus=corpus)
        speakers = list(utts["target_child_name"].unique())
        path = out_path / f'{collection}'

        if separate_by_child:
            path = path / f'{corpus}'
            if not path.exists():
                path.mkdir(parents=True)
            for speaker in speakers:
                a = utts[utts["target_child_name"] == speaker]
                out_path = path /f'{speaker}.csv'
                if out_path.exists():
                    out_path.unlink()
                self.logger.info(f'Saving {len(a)} utterances to {out_path}')
                a.to_csv(out_path)
        else:
            if not path.exists():
                path.mkdir(parents=True)
            out_path = path / f'{collection if corpus is None else corpus}.csv'
            utts.to_csv(out_path)
            self.logger.info(f'Saving {len(utts)} utterances to {out_path}')
