# pylint: disable=unused-import, import-outside-toplevel


class TestGrobidImports:
    def test_can_import_sequence(self):
        from sciencebeam_trainer_delft.grobid.sequenceLabelling import Sequence  # noqa

    def test_can_import_load_data_and_labels_crf_file(self):
        from sciencebeam_trainer_delft.grobid.sequenceLabelling.reader import (  # noqa
            load_data_and_labels_crf_file
        )

    def test_can_import_load_data_crf_string(self):
        from sciencebeam_trainer_delft.grobid.sequenceLabelling.reader import (  # noqa
            load_data_crf_string
        )

    def test_can_import_embeddings(self):
        from sciencebeam_trainer_delft.grobid.utilities.Embeddings import (  # noqa
            Embeddings
        )
