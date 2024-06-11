#!/usr/bin/env bash
# docker run -v .:/testfun ghcr.io/darpa-askem/funman-taskrunner:latest /testfun/test.sh

FUNMAN_IMAGE=ghcr.io/darpa-askem/funman-taskrunner:latest 

docker run -v .:/testfun ${FUNMAN_IMAGE} /bin/bash -c "cd /testfun && python fun.py"
