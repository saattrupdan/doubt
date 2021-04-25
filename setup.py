from setuptools import setup, find_packages
from pathlib import Path
from bump_version import get_current_version


setup(name='doubt',
      version=get_current_version(return_tuple=False),
      description='Bringing back uncertainty to machine learning',
      long_description=Path('README.md').read_text(),
      long_description_content_type='text/markdown',
      url='https://github.com/saattrupdan/doubt',
      author='Dan Saattrup Nielsen',
      author_email='saattrupdan@gmail.com',
      license='MIT',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'],
      packages=find_packages(exclude=('tests',)),
      include_package_data=True,
      install_requires=['numpy>=1.20',
                        'scipy>=1.6',
                        'scikit-learn>=0.24',
                        'requests>=2.25',
                        'PyYAML>=5.4',
                        'tables>=3.6',
                        'xlrd>=2.0',
                        'openpyxl>=3.0'])
