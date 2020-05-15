import logging
import os
from hashlib import md5
from pathlib import Path

from sciencebeam_trainer_delft.utils.io import (
    copy_file,
    is_external_location,
    strip_compression_filename_ext
)


LOGGER = logging.getLogger(__name__)


DEFAULT_DOWNLOAD_DIR = 'data/download'


class DownloadManager:
    def __init__(self, download_dir: str = DEFAULT_DOWNLOAD_DIR):
        self.download_dir = Path(download_dir)

    def get_local_file(self, file_url: str, auto_uncompress: bool = True) -> str:
        filename = '%s-%s' % (
            md5(file_url.encode('utf-8')).hexdigest(),
            os.path.basename(file_url)
        )
        if auto_uncompress:
            filename = strip_compression_filename_ext(filename)
        return str(self.download_dir.joinpath(filename))

    def is_downloaded(self, file_url: str, auto_uncompress: bool = True) -> bool:
        download_file = self.get_local_file(file_url, auto_uncompress=auto_uncompress)
        return os.path.exists(download_file)

    def download(
            self, file_url: str,
            local_file: str = None,
            auto_uncompress: bool = True,
            skip_if_downloaded: bool = True) -> str:
        download_file = (
            local_file or self.get_local_file(file_url, auto_uncompress=auto_uncompress)
        )
        if skip_if_downloaded and self.is_downloaded(file_url, auto_uncompress=auto_uncompress):
            LOGGER.info('file already downloaded: %s (%s)', file_url, download_file)
        else:
            LOGGER.info('copying %s to %s', file_url, download_file)
            copy_file(file_url, download_file)
        return download_file

    def download_if_url(self, file_url_or_path: str, **kwargs) -> str:
        if is_external_location(file_url_or_path):
            return self.download(file_url_or_path, **kwargs)
        return file_url_or_path
