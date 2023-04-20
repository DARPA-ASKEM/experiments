# Terarium Glossary

Here, we define and contrast common terms used to label different concepts within Terarium.

## Model | Simulation

* A `model` is an abstract representation of a system with the purpose of approximating its behaviours,
e.g. an epidemic can be approximated by a model built from a set of ordinary differential equations.

* A `simulation` is an instance of a model that is executed to accept inputs and generate outputs with the goal of approximating the behaviours of the underlying system under different conditions.

* "Modeling" is the process of building a model and a "simulator" is an agent that takes a corresponding model and some input values and generates output values.

## Scenarios | Configurations | Runs

* A `scenario` is a natural-language description of the context, problems, or questions that is the starting point of the modeling and simulation process.

* A `configuration` is any set of values that can be used as input for a given model; it is a model-specific representation of a scenario.

* A `run` is the output of a simulation.

* Given a system and scenario, a (model, configuration) pair can be constructed and executed by a simulator to initiate simulations that generate each a run.

## State Variables | Parameters | Hyperparameters | Initial Conditions

* A `state variable` is a varying quantity of a given system (and corresponding model) that, in combination of others, can fully determine the "state" of the underlying system,
e.g. pressure, volume, temperature are the state variables of an ideal gas.

* A `parameter` is a fixed quantity of a given model and consists of the inputs and constants internal to the model; they can be inferred from data (observations of the underlying system),
e.g. weights and biases are parameters of an artificial neural network model.

* A `hyperparameter` is a fixed quantity that is an input of the simulator; they cannot be inferred from data and can impact the precision and accuracy of the resulting simulation,
e.g. the polynomial degree of a regression model and the learning rate of an optimization algorithm.

## Observables | Observation Functions

* An `observable` is a quantity of a given system (and corresponding model) that can be measured as "observation" data points.

* An `observation function` is a function that maps from the state variables to an observable, capturing the physics of the measurement process, 
e.g. a epidemiological model has a state variable named "total infected population" and has "confirmed number of infections" as an observable 
and the mathematical expression relating the two quantities is the observation function.

## Fitting | Calibration | Optimization

* `Fitting` and `calibration` are equivalent terms that describe, given a model and a set of observations of the underlying system, the process of determining for model parameter values 
that yields the least approximation errors when used as inputs into a simulation of the model.

* `Optimization` is the process of determining the values of a function that best meet some constraints; 
fitting and calibration are particular cases where the constraint is the approximation error between the model outputs and the system observations.

## Interventions | Assumptions


## Workflow Graphs | Provenance Graphs | Lineage Graphs
