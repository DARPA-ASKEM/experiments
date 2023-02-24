FROM "jupyter/minimal-notebook"

USER root

ENV JULIA_VERSION=1.7.3

RUN mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz

RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

USER $NB_UID

# Add python packages
RUN pip3 install numpy \
    scipy \
    pandas \
    matplotlib \
    jupyterlab \
    networkx \
    scikit-learn \
    tqdm \
    pylint \
    regex \
    requests
    minisom \
    tslearn

# Add packages and precompile
RUN julia -e 'import Pkg; Pkg.update()' && \
    julia -e 'import Pkg; Pkg.add("Colors"); using Colors' && \
    julia -e 'import Pkg; Pkg.add("Catlab"); using Catlab' && \
    julia -e 'import Pkg; Pkg.add("IJulia"); using IJulia' && \
    julia -e 'import Pkg; Pkg.add("GraphViz"); using GraphViz' && \
    julia -e 'import Pkg; Pkg.add("AlgebraicPetri"); using AlgebraicPetri' && \
    julia -e 'import Pkg; Pkg.add("DifferentialEquations"); using DifferentialEquations' && \
    julia -e 'import Pkg; Pkg.add("Plots"); using Plots' && \
    julia -e 'import Pkg; Pkg.add("JSON"); using JSON' && \
    julia -e 'import Pkg; Pkg.add("LabelledArrays"); using LabelledArrays' && \
    julia -e 'import Pkg; Pkg.add("PrettyTables"); using PrettyTables' && \
    julia -e 'import Pkg; Pkg.add("OrdinaryDiffEq"); using OrdinaryDiffEq' && \
    julia -e 'import Pkg; Pkg.add("ModelingToolkit"); using ModelingToolkit' && \
    fix-permissions /home/$NB_USER

CMD jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password=''
