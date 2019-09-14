import json

import flair.datasets
from dataclasses import dataclass

from flair import logger
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import (
    FlairEmbeddings,
    StackedEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from pathlib import Path

from typing import Dict, List, Tuple, Union


@dataclass
class Experiment:
    description: str
    language: str
    language_code: str
    conll_path: str
    embeddings: List[str]
    layers: List[int]
    batch_size: int
    hidden_size: int
    max_epochs: int
    embeddings_storage_mode: str
    pooling_operation: str
    use_crf: bool
    use_scalar_mix: bool
    train_with_dev: bool


class ExperimentRunner:
    def __init__(self, number: int, configuration_file: str):
        self.experiment = Experiment(**self._get_experiment_details(configuration_file))
        self.number = number

        logger.info(self.experiment)

    def start(self) -> None:
        self.stacked_embeddings = self._get_stacked_embeddings()
        description = self.experiment.description.replace(" ", "_")
        batch_size = self.experiment.batch_size
        max_epochs = self.experiment.max_epochs
        embeddings_storage_mode = self.experiment.embeddings_storage_mode
        train_with_dev = self.experiment.train_with_dev

        tagger, corpus = self._get_sequence_tagger()

        trainer = ModelTrainer(tagger, corpus)

        trainer.train(
            f"resources/taggers/experiment_{description}_{self.number}",
            learning_rate=0.1,
            mini_batch_size=batch_size,
            max_epochs=max_epochs,
            embeddings_storage_mode=embeddings_storage_mode,
            train_with_dev=train_with_dev,
        )

    def _get_experiment_details(
        self, configuration_file: str
    ) -> Dict[str, Union[str, bool, int]]:
        with open(configuration_file, "r") as f_p:
            return json.load(f_p)

    def _get_stacked_embeddings(self) -> StackedEmbeddings:
        token_embeddings = []

        for embedding in self.experiment.embeddings:
            language_or_path = "-".join(embedding.split("-")[1:])
            if embedding.startswith("word"):
                logger.info(f"Using {language_or_path} as Word Embeddings")
                token_embeddings.append(WordEmbeddings(language_or_path))
            elif embedding.startswith("flair"):
                logger.info(f"Using {language_or_path} as Flair Embeddings")
                token_embeddings.append(FlairEmbeddings(language_or_path))

        return StackedEmbeddings(embeddings=token_embeddings)

    def _get_sequence_tagger(self) -> Tuple[SequenceTagger, UniversalDependenciesCorpus]:
        language = self.experiment.language
        language_code = self.experiment.language_code
        conll_path = self.experiment.conll_path
        corpus = UniversalDependenciesCorpus(data_folder=Path(f'./'),
                                         train_file=Path(f'{conll_path}/UD_{language}/{language_code}-ud-train.conllu'),
                                         dev_file=Path(f'{conll_path}/UD_{language}/{language_code}-ud-dev.conllu'),
                                         test_file=Path(f'{conll_path}/UD_{language}/{language_code}-ud-test.conllu'))

        tag_type = "upos"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        tagger = SequenceTagger(
            hidden_size=self.experiment.hidden_size,
            embeddings=self.stacked_embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=self.experiment.use_crf,
        )

        return tagger, corpus
