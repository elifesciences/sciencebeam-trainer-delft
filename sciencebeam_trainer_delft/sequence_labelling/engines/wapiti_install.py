import logging
import tarfile
import os
import subprocess
from typing import Optional

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


LOGGER = logging.getLogger(__name__)


TAR_GZ_EXT = '.tar.gz'


def install_wapiti_and_get_path_or_none(
    install_url: Optional[str],
    download_manager: DownloadManager
) -> Optional[str]:
    if not install_url:
        return None
    if not install_url.endswith(TAR_GZ_EXT):
        raise ValueError('only supporting %s' % TAR_GZ_EXT)
    local_file = download_manager.download_if_url(
        install_url,
        auto_uncompress=False
    )
    extracted_directory = local_file[:-len(TAR_GZ_EXT)]
    LOGGER.debug('local_file: %s', local_file)
    LOGGER.debug('extracting to: %s', extracted_directory)
    with tarfile.open(local_file, mode='r') as tar:
        tar.extractall(extracted_directory)
    extracted_files = os.listdir(extracted_directory)
    wapiti_source_directory = extracted_directory
    if len(extracted_files) == 1:
        wapiti_source_directory = os.path.join(extracted_directory, extracted_files[0])
    LOGGER.info('running make in %s', wapiti_source_directory)
    subprocess.check_output(
        'make',
        cwd=wapiti_source_directory
    )
    wapiti_binary = os.path.join(wapiti_source_directory, 'wapiti')
    LOGGER.info('done, binary: %s', wapiti_binary)
    return wapiti_binary
