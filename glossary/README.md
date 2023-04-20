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

* A `state variable` is a varying quantity of a given system (and corresponding model) that, in combination of others, can fully determine the "state" of the underlying system;
e.g. `S`, `I`, `R` are the state variables of the SIR compartmental model.

* A `parameter` is a fixed quantity of a given model and consists of the inputs and constants internal to the model; they can be inferred from data (observations of the underlying system);
e.g. `β`, `γ` of the SIR compartmental model and weights of an artificial neural network model.

* A `hyperparameter` is a fixed quantity that is an input of the simulator; 
they cannot be inferred from data and can impact the precision and accuracy of the resulting simulation;
e.g. `loss`, `penalty`, `tol`, etc. of the [stochastic gradient descent algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor).

* An `initial condition` is a parameter that corresponds to the value of a state variable at the starting time point; in a given model, there are as many initial conditions as state variables;
e.g. `S₀`, `I₀`, `R₀` are the initial conditions of the SIR compartmental model.

## Observables | Observation Functions | Alignments

* An `observable` is a quantity of a given system (and corresponding model) that can be measured as "observation" data points; e.g. `I_obs` (observed infected population), `N` (total population), `R_frac` (recovered population fraction), `ℜ₀` ([basic reproduction ratio](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3157160/)), and `inc_I_obs` (observed incident infection rate) can be observables of the SIR compartmental model. 

* An `observation function` is a function that maps state variables (and observables) to a given observable, capturing knowledge such as the physics of the observation or measurement process and expert heuristics;
e.g. `I_obs = 0.50 * I`, `N = S + I + R`, `R_frac = R / N`, `ℜ₀ = β * S / γ`, `inc_I_ob = diff(I_obs(t), t)) * Heaviside(diff(I_obs(t), t))`.

* An `alignment` is a one-to-one mapping between quantities of a given model and features of a given dataset that enables simulations such model calibration;
e.g. assuming the SIR compartmental model and a training dataset with features `truth-incident_cases`, `truth-incident_deaths`, `truth-incident_hospitalization`, we can have the following model-data alignment:
```json
{
    "inc_I_obs": "truth-incident_cases",
    "inc_D": "truth-incident_deaths",
    "inc_H": "truth-incident_hospitalization"
}
```

* Note that "incident" refers to *new* occurrences, as opposed to "prevalent" which refers to *current* (new and pre-existing) occurrences.

## Fitting | Training | Calibration | Optimization

* `Fitting`, `training`, and `calibration` are equivalent terms that describe, given a model and a set of observations of the underlying system, the process of determining for model parameter values 
that yields the least approximation errors when used as inputs into a simulation of the model.

* `Optimization` is the process of determining the values of a function that best meet some constraints; 
fitting and calibration are particular cases where the constraint is the approximation error between the model outputs and the system observations.

## Interventions | Assumptions



## Workflow Graphs | Provenance Graphs | Lineage Graphs


