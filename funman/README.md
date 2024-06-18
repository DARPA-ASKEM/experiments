# FUNMAN

`funman` (Functional Model Analysis) is a tool developed to segment the input parameter space of a model and classify the segments into `True/False` regions, depending on whether 
the resulting model outputs satisfy some given set of constraints.

The main repo is (here)[https://github.com/DARPA-ASKEM/funman-api].

To run:
```shell
docker pull ghcr.io/darpa-askem/funman-taskrunner:latest
./run.sh > result.json
```