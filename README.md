# Spatio-temporal analysis of B.1.1.7 infection dynamics

This repository contains code to analyse strain-specific infection dynamics based on generic PCR test and B.1.1.7 genomic prevalence data across 382 British lower tier local authorities (LTLAs).

The code was developed by Harald Vöhringer and amended by Moritz Gerstung.

Data is from publicly available sources and COG-UK.

The analysis is implemented in `python` and uses `numpyro` for stochastic variation inference of the underlying model.

You can install the necessary python packages using (tested with `pip` version 20.2.3):

```
$ pip install -r requirements.txt
```

Additionally install `numpyro` with the `--no-deps` flag.

```
pip install -r requirements-no-deps.txt --no-deps
```

Depending on your installation you may need to install libspatialindex using

```
$ brew install spatialindex
```

or

```
$ conda install -c conda-forge libspatialindex
```

The main notebook is `sarscov2-b117.ipynb`
