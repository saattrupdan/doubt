from setuptools import setup, find_packages


setup(name='doubt',
      maintainer='Dan Saattrup Nielsen',
      maintainer_email='saattrupdan@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      description='Bringing back uncertainty to machine learning',
      license='new BSD',
      url='https://github.com/saattrupdan/doubt',
      version='beta',
      zip_safe=False,
      install_requires=['numpy',
                        'scipy',
                        'scikit-learn',
                        'requests',
                        'PyYAML',
                        'tables',
                        'openpyxl'])
