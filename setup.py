from __future__ import print_function

import os
import subprocess
import shlex

from distutils.command.build import build  # pylint: disable=import-error, no-name-in-module

from setuptools import (
    find_packages,
    setup,
    Command
)

with open(os.path.join('requirements.txt'), 'r') as f:
    REQUIRED_PACKAGES = f.readlines()

with open(os.path.join('requirements.delft.txt'), 'r') as f:
    DELFT_PACKAGES = f.readlines()


def _run_command(command):
    command_args = shlex.split(command)
    print('Running command: %s' % command_args)
    p = subprocess.Popen(
        command_args,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
        raise RuntimeError(
            'Command %s failed: exit code: %s (output: %s)' %
            (command_args, p.returncode, stdout_data)
        )


def _is_delft_installed():
    try:
        import delft  # pylint: disable=unused-import
        return  True
    except ImportError:
        return False


def _install_delft():
    _run_command('pip3 install --no-deps %s' % ' '.join(DELFT_PACKAGES))


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
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    include_package_data=True,
    description='ScienceBeam Trainer DeLFT',
    cmdclass={
        'build': CustomBuild,
        'CustomCommands': CustomCommands
    }
)
