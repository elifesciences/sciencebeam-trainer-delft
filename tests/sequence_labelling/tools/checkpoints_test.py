import json
from pathlib import Path

from sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints import (
    main
)


def _create_checkpoint(path: Path, meta: dict):
    path.mkdir(exist_ok=True)
    path.joinpath('meta.json').write_text(json.dumps(meta))
    return path


class TestMain:
    def test_should_format_as_json(
            self, temp_dir: Path, capsys):
        epoch1_path = _create_checkpoint(
            temp_dir.joinpath('epoch-1'),
            {'f1': 0.1}
        )
        epoch2_path = _create_checkpoint(
            temp_dir.joinpath('epoch-2'),
            {'f1': 0.2}
        )
        epoch3_path = _create_checkpoint(
            temp_dir.joinpath('epoch-3'),
            {'f1': 0.15}
        )
        temp_dir.joinpath('checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': str(epoch1_path)
            }, {
                'path': str(epoch2_path)
            }, {
                'path': str(epoch3_path)
            }],
            'last_checkpoint': {
                'path': str(epoch3_path)
            }
        }))
        main([
            '--output-format=json',
            '--checkpoint=%s' % temp_dir
        ])
        out, _ = capsys.readouterr()
        result_list = json.loads(out)
        assert result_list
        assert [
            item['path']
            for item in result_list
        ] == [
            str(epoch1_path),
            str(epoch3_path),
            str(epoch2_path)
        ]
