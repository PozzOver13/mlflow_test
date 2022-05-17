from setuptools import setup, find_packages

config = {
    'name': 'mlflow_test',
    'description': 'mlflow test',
    'author': 'ste',
    'packages': find_packages()
}

setup(**config)