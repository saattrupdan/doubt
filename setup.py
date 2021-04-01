from setuptools import setup, find_packages
from pathlib import Path


setup(name='doubt',
      version='1.0.0',
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
      install_requires=['numpy', 'scipy', 'scikit-learn', 'requests',
                        'PyYAML', 'tables', 'openpyxl'])
