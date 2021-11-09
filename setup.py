from __future__ import print_function

import os
import subprocess
import sys

from distutils.command.build import build  # type: ignore

from setuptools import (
    find_packages,
    setup,
    Command
)

import sciencebeam_trainer_delft


with open(os.path.join('requirements.txt'), 'r') as f:
    REQUIRED_PACKAGES = f.readlines()

with open(os.path.join('requirements.delft.txt'), 'r') as f:
    DELFT_PACKAGES = f.readlines()

with open('README.md', 'r') as f:
    long_description = f.read()


def _run_command(command_args):
    print('Running command: %s' % command_args)
    with subprocess.Popen(
        command_args,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        stdout_data, _ = process.communicate()
        print('Command output: %s' % stdout_data)
        if process.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s (output: %s)' %
                (command_args, process.returncode, stdout_data)
            )


def _is_delft_installed():
    try:
        import delft  # noqa pylint: disable=unused-import, import-outside-toplevel
        return True
    except ImportError:
        return False


def _install_delft():
    _run_command(
        [sys.executable, '-m', 'pip', 'install', '--no-deps']
        + DELFT_PACKAGES
    )


def _install_delft_if_not_installed():
    if _is_delft_installed():
        print('delft already installed, skipping')
    else:
        _install_delft()


class CustomCommands(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _install_delft_if_not_installed()


class CustomBuild(build):
    """A build command class that will be invoked during package install.
    The package built using the current setup.py will be staged and later
    installed in the worker using `pip install package'. This class will be
    instantiated during install for this specific scenario and will trigger
    running the custom commands specified.
    """
    sub_commands = build.sub_commands + [('CustomCommands', None)]


packages = find_packages()

setup(
    name='sciencebeam_trainer_delft',
    version=sciencebeam_trainer_delft.__version__,
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    include_package_data=True,
    description='ScienceBeam Trainer DeLFT',
    cmdclass={
        'build': CustomBuild,
        'CustomCommands': CustomCommands
    },
    url='https://github.com/elifesciences/sciencebeam-trainer-delft',
    license='MIT',
    keywords="sciencebeam delft",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
