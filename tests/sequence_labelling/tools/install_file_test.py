import gzip
from pathlib import Path

from sciencebeam_trainer_delft.sequence_labelling.tools.install_file import main


BINARY_DATA_1 = b'test binary data 1'
BINARY_DATA_2 = b'test binary data 2'


class TestMain:
    def test_should_copy_file(self, tmp_path: Path):
        source_file_path = tmp_path / 'source.bin'
        source_file_path.write_bytes(BINARY_DATA_1)
        target_file_path = tmp_path / 'target.bin'
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}'
        ])
        assert target_file_path.exists()
        assert target_file_path.read_bytes() == BINARY_DATA_1

    def test_should_not_replace_file_if_source_url_matches(self, tmp_path: Path):
        source_file_path = tmp_path / 'source.bin'
        source_file_path.write_bytes(BINARY_DATA_1)
        target_file_path = tmp_path / 'target.bin'
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}'
        ])
        source_file_path.write_bytes(BINARY_DATA_2)
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}'
        ])
        assert target_file_path.exists()
        assert target_file_path.read_bytes() == BINARY_DATA_1

    def test_should_replace_file_if_force_flag_was_used(self, tmp_path: Path):
        source_file_path = tmp_path / 'source.bin'
        source_file_path.write_bytes(BINARY_DATA_1)
        target_file_path = tmp_path / 'target.bin'
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}'
        ])
        source_file_path.write_bytes(BINARY_DATA_2)
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}',
            '--force'
        ])
        assert target_file_path.exists()
        assert target_file_path.read_bytes() == BINARY_DATA_2

    def test_should_gunzip_file(self, tmp_path: Path):
        source_file_path = tmp_path / 'source.bin.gz'
        source_file_path.write_bytes(
            gzip.compress(BINARY_DATA_1)
        )
        target_file_path = tmp_path / 'target.bin'
        main([
            f'--source={source_file_path}',
            f'--target={target_file_path}'
        ])
        assert target_file_path.exists()
        assert target_file_path.read_bytes() == BINARY_DATA_1
