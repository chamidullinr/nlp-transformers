# NLP Transformers

A simple Python library build on top of Hugging Face `transformers` library.


## Getting Started
The library is developed and tested on `Python=3.7`.

The library uses packages `torch`, `transformers`, `datasets`, `numpy`, `pandas`, and `scikit-learn`.
All dependencies including versions are listed in [requirements.txt](requirements.txt).
To install required packages, run:
```bash
pip install -r requirements.txt
```


## Build the Package
Create a wheel file that can be installed through ``pip`` package manager:
``` bash
python setup.py bdist_wheel
```

Then, the package can be installed as easy as
``` bash
pip install dist/[GENERATED_FILE_NAME].whl
```

## Authors
**Ray Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)
