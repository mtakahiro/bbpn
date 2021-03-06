import re
import sys
import os
from pkg_resources import get_distribution, DistributionNotFound

__version_commit__ = ''
_regex_git_hash = re.compile(r'.*\+g(\w+)')

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'dev'

if '+' in __version__:
    commit = _regex_git_hash.match(__version__).groups()
    if commit:
        __version_commit__ = commit[0]

__author__ = 'Takahiro Morishita'
__email__ = 'takahiro@ipac.caltech.edu'
__credits__ = 'IPAC'

package = 'bbpn'

print('Welcome to %s version %s'%(package,__version__))

