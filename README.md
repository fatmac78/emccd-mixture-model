# EMCCD Mixture Model

## Introduction

Like the name suggests, this library can be used to quickly fit a mixture model of an 
the output distribution of an EMCCD to some data. This model incorporates the effects
of readout noise, parallel clock induced charge and serial clock induced charge.  
Expectation Maximization if used to fit the model to the data. The library is a modified 
version of [this](https://github.com/ethan-homan/gaussian-exponential-mixture), which models 
a simpler mixture of just a Gaussian and exponential

## License

The code is licensed under the Apache 2.0 License. 

## Installing

This requires python 3.6 +

```shell script
git clone https://github.com/fatmac78/emccd-mixture-model.git
cd emccd-mixture-model
pip install -r requirements.txt
```

## Usage

TO DO
