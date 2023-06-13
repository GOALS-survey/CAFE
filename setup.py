from setuptools import setup, find_packages

setup(
    name='CAFE',
    version='1.0.1',    
    url='',
    author='Thomas Lai, Tanio Diaz-Santos, Luke Finnerty, and the GOALS Team',
    author_email='goals.lirg.survey',
    license='GPL-3',
    long_description='Astronomical spectroscopy decomposition.',

    packages=find_packages(),

    include_package_data = True,

    data_files=[
        ('./CAFE', []),
        ('./CRETA/data', []),
    ],

    install_requires=[
        'astropy>=4.3.1',
        'lmfit>=1.0.3',
        'matplotlib>=3.5.1',
        'numpy>=1.21.5',
        'pandas>=1.3.5',
        'photutils>=1.5.0',
        'scipy>=1.7.3',
        'specutils>1.7.0',
        'ipykernel>6.9.1'
    ]
)
