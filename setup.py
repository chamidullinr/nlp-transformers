from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = ''.join(f.readlines())

setup(
    name='nlp-transformers',
    version='0.2.3',
    description='A Python library for applying Transformers on various NLP tasks.',
    long_description=long_description,
    py_module=['nlp_transformers'],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1.19',
        'pandas >= 1.1',
        'scikit-learn >= 0.24',
        # 'torch >= 1.9',
        'transformers >= 4.10',
        'datasets >= 1.12'
    ],
    zip_safe=False
)
