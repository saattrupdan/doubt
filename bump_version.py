'''Script that fetches and bumps versions'''

from pathlib import Path
import re
from typing import Union, Tuple
import subprocess


def get_current_version(return_tuple: bool = False) \
                        -> Union[str, Tuple[int, int, int]]:
    '''Fetch the current version without import __init__.py.

    Args:
        return_tuple (bool, optional):
            Whether to return a tuple of three numbers, corresponding to the
            major, minor and patch version. Defaults to False.

    Returns:
        str or tuple of three ints: The current version
    '''
    init_file = Path('doubt') / '__init__.py'
    init = init_file.read_text()
    version_regex = r"(?<=__version__ = ')[0-9]+\.[0-9]+\.[0-9]+(?=')"
    version = re.search(version_regex, init)[0]
    if return_tuple:
        major, minor, patch = [int(v) for v in version.split('.')]
        return (major, minor, patch)
    else:
        return version


def set_new_version(major: int, minor: int, patch: int):
    '''Sets a new version.

    Args:
        major (int):
            The major version. This only changes when the code stops being
            backwards compatible.
        minor (int):
            The minor version. This changes when a backwards compatible change
            happened.
        patch (init):
            The patch version. This changes when the only new changes are bug
            fixes.
    '''
    # Get current __init__.py content
    init_file = Path('doubt') / '__init__.py'
    init = init_file.read_text()

    # Replace __version__ in __init__.py with the new one
    version_regex = r"(?<=__version__ = ')[0-9]+\.[0-9]+\.[0-9]+(?=')"
    new_init = re.sub(version_regex, f'{major}.{minor}.{patch}', init)
    with init_file.open('w') as f:
        f.write(new_init)

    # Add to version control
    subprocess.run(['git', 'add', 'doubt/__init__.py'])
    subprocess.run(['git', 'commit', '-m', f'feat: v{major}.{minor}.{patch}'])


def create_version_tag():
    '''Creates a tag of the form v[MAJOR].[MINOR].[PATCH]'''
    version = get_current_version(return_tuple=False)
    subprocess.run(['git', 'tag', f'v{version}'])


def bump_major():
    '''Add one to the major version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major + 1, 0, 0)
    create_version_tag()


def bump_minor():
    '''Add one to the minor version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major, minor + 1, 0)
    create_version_tag()


def bump_patch():
    '''Add one to the patch version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major, minor, patch + 1)
    create_version_tag()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--major', const=True, nargs='?', default=False,
                        help='Bump the major version by one.')
    parser.add_argument('--minor', const=True, nargs='?', default=False,
                        help='Bump the minor version by one.')
    parser.add_argument('--patch', const=True, nargs='?', default=False,
                        help='Bump the patch version by one.')
    args = parser.parse_args()

    if args.major + args.minor + args.patch != 1:
        raise RuntimeError('Exactly one of --major, --minor and --patch must '
                           'be selected.')
    elif args.major:
        bump_major()
    elif args.minor:
        bump_minor()
    elif args.patch:
        bump_patch()
