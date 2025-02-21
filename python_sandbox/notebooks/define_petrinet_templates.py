# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # Define AMR of Template Models for PetriNet Framework
# 
# We define template models as the "basis vectors" of the PetriNet framework: 
# any PetriNet model should be a linear combination of such template models. 
# 
# Per the [MIRA package](https://github.com/gyorilab/mira/blob/main/mira/metamodel/templates.py):
# 1. natural conversion
# 2. natural production
# 3. natural degradation
# 4. controlled conversion
# 5. controlled production
# 6. controlled degradation
# 7. observable 

# %%
import os
import json

# %%
model_mmt = {"templates": [], "parameters": [], "initials": {}}

# %%
# Default parameter and initial condition values

model_mmt["parameters"] = {
    "alpha": {
        "name": "alpha",
        "identifiers": {},
        "context": {},
        "value": 0.57
    },
    "beta": {
        "name": "beta",
        "identifiers": {},
        "context": {},
        "value": 0.011
    },
    "gamma": {
        "name": "gamma",
        "identifiers": {},
        "context": {},
        "value": 0.456
    },
    "delta": {
        "name": "delta",
        "identifiers": {},
        "context": {},
        "value": 0.011
    },
    "epsilon": {
        "name": "epsilon",
        "identifiers": {},
        "context": {},
        "value": 0.171
    },
    "theta": {
        "name": "theta",
        "identifiers": {},
        "context": {},
        "value": 0.371
    },
    "zeta": {
        "name": "zeta",
        "identifiers": {},
        "context": {},
        "value": 0.125
    },
    "eta": {
        "name": "eta",
        "identifiers": {},
        "context": {},
        "value": 0.125
    },
    "mu": {
        "name": "mu",
        "identifiers": {},
        "context": {},
        "value": 0.017
    },
    "nu": {
        "name": "nu",
        "identifiers": {},
        "context": {},
        "value": 0.027
    },
    "tau": {
        "name": "tau",
        "identifiers": {},
        "context": {},
        "value": 0.01
    },
    "kappa": {
        "name": "kappa",
        "identifiers": {},
        "context": {},
        "value": 0.017
    },
    "rho": {
        "name": "rho",
        "identifiers": {},
        "context": {},
        "value": 0.034
    },
    "sigma": {
        "name": "sigma",
        "identifiers": {},
        "context": {},
        "value": 0.017
    },
    "xi": {
        "name": "xi",
        "identifiers": {},
        "context": {},
        "value": 0.017
    },
    "Event_trigger_Fig3b": {
        "name": "Event_trigger_Fig3b",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "Event_trigger_Fig3d": {
        "name": "Event_trigger_Fig3d",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "Event_trigger_Fig4b": {
        "name": "Event_trigger_Fig4b",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "Event_trigger_Fig4d": {
        "name": "Event_trigger_Fig4d",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "epsilon_modifier": {
        "name": "epsilon_modifier",
        "identifiers": {},
        "context": {},
        "value": 1.0
    },
    "alpha_modifier": {
        "name": "alpha_modifier",
        "identifiers": {},
        "context": {},
        "value": 1.0
    },
    "ModelValue_16": {
        "name": "ModelValue_16",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "ModelValue_17": {
        "name": "ModelValue_17",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "ModelValue_18": {
        "name": "ModelValue_18",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "ModelValue_19": {
        "name": "ModelValue_19",
        "identifiers": {},
        "context": {},
        "value": 0.0
    },
    "ModelValue_21": {
        "name": "ModelValue_21",
        "identifiers": {},
        "context": {},
        "value": 1.0
    },
    "ModelValue_20": {
        "name": "ModelValue_20",
        "identifiers": {},
        "context": {},
        "value": 1.0
    },
    "Italy": {
        "name": "Italy",
        "identifiers": {},
        "context": {},
        "value": 1.0
    },
    "XXlambdaXX": {
        "name": "XXlambdaXX",
        "identifiers": {},
        "context": {},
        "value": 0.034
    }
}

model_mmt["initials"] =  {
    "Susceptible": 0.9999963,
    "Infected": 3.33333333e-06,
    "Diagnosed": 3.33333333e-07,
    "Ailing": 1.66666666e-08,
    "Recognized": 3.33333333e-08,
    "Threatened": 0.0,
    "Healed": 0.0,
    "Extinct": 0.0
}

# %%
# Templates
model_mmt["templates"] = [
    # 1.  S -> alpha (I) -> I
    {
        "rate_law": "Susceptible*Infected*alpha",
        "type": "ControlledConversion",
        "controller": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "subject": {
            "name": "Susceptible",
            "identifiers": {
                "ido": "0000514",
                "biomodels.species": "BIOMD0000000955:Susceptible"
            },
            "context": {
                "property": "ido:0000468"
            }
        },
        "outcome": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "provenance": []
    },
    # 2.  S -> beta (D)  -> I
    {
        "rate_law": "Susceptible*Diagnosed*beta",
        "type": "ControlledConversion",
        "controller": {
            "name": "Diagnosed",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Diagnosed"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "subject": {
            "name": "Susceptible",
            "identifiers": {
                "ido": "0000514",
                "biomodels.species": "BIOMD0000000955:Susceptible"
            },
            "context": {
                "property": "ido:0000468"
            }
        },
        "outcome": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "provenance": []
    },
    # 3.  S -> gamma (A) -> I
    {
        "rate_law": "Susceptible*Ailing*gamma",
        "type": "ControlledConversion",
        "controller": {
            "name": "Ailing",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Ailing"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "subject": {
            "name": "Susceptible",
            "identifiers": {
                "ido": "0000514",
                "biomodels.species": "BIOMD0000000955:Susceptible"
            },
            "context": {
                "property": "ido:0000468"
            }
        },
        "outcome": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "provenance": []
    },
    # 4.  S -> delta (R) -> I
    {
        "rate_law": "Susceptible*Recognized*delta",
        "type": "ControlledConversion",
        "controller": {
            "name": "Recognized",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Recognized"
            },
            "context": {
                "property": "ncit:C25587"
            }
        },
        "subject": {
            "name": "Susceptible",
            "identifiers": {
                "ido": "0000514",
                "biomodels.species": "BIOMD0000000955:Susceptible"
            },
            "context": {
                "property": "ido:0000468"
            }
        },
        "outcome": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "provenance": []
    },
    # 5.  I -> epsilon   -> D
    {
        "rate_law": "1.0*Infected*epsilon",
        "type": "NaturalConversion",
        "subject": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Diagnosed",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Diagnosed"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "provenance": []
    },
    # 6.  I -> zeta      -> A
    {
        "rate_law": "1.0*Infected*zeta",
        "type": "NaturalConversion",
        "subject": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Ailing",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Ailing"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "provenance": []
    },
    # 7.  I -> lambda    -> H
    {
        "rate_law": "1.0*Infected*XXlambdaXX",
        "type": "NaturalConversion",
        "subject": {
            "name": "Infected",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Infected"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Healed",
            "identifiers": {
                "biomodels.species": "BIOMD0000000955:Healed"
            },
            "context": {
                "property": "ido:0000621"
            }
        },
        "provenance": []
    },
    # 8.  D -> eta       -> R
    {
        "rate_law": "1.0*Diagnosed*eta",
        "type": "NaturalConversion",
        "subject": {
            "name": "Diagnosed",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Diagnosed"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "outcome": {
            "name": "Recognized",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Recognized"
            },
            "context": {
                "property": "ncit:C25587"
            }
        },
        "provenance": []
    },
    # 9.  D -> rho       -> H
    {
        "rate_law": "1.0*Diagnosed*rho",
        "type": "NaturalConversion",
        "subject": {
            "name": "Diagnosed",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Diagnosed"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "outcome": {
            "name": "Healed",
            "identifiers": {
                "biomodels.species": "BIOMD0000000955:Healed"
            },
            "context": {
                "property": "ido:0000621"
            }
        },
        "provenance": []
    },
    # 10. A -> theta     -> R
    {
        "rate_law": "1.0*Ailing*theta",
        "type": "NaturalConversion",
        "subject": {
            "name": "Ailing",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Ailing"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Recognized",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Recognized"
            },
            "context": {
                "property": "ncit:C25587"
            }
        },
        "provenance": []
    },
    # 11. A -> kappa     -> H 
    {
        "rate_law": "1.0*Ailing*kappa",
        "type": "NaturalConversion",
        "subject": {
            "name": "Ailing",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Ailing"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Healed",
            "identifiers": {
                "biomodels.species": "BIOMD0000000955:Healed"
            },
            "context": {
                "property": "ido:0000621"
            }
        },
        "provenance": []
    },
    # 12. A -> mu        -> T
    {
        "rate_law": "1.0*Ailing*mu",
        "type": "NaturalConversion",
        "subject": {
            "name": "Ailing",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Ailing"
            },
            "context": {
                "property": "ncit:C113725"
            }
        },
        "outcome": {
            "name": "Threatened",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Threatened"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "provenance": []
    },
    # 13. R -> nu        -> T
    {
        "rate_law": "1.0*Recognized*nu",
        "type": "NaturalConversion",
        "subject": {
            "name": "Recognized",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Recognized"
            },
            "context": {
                "property": "ncit:C25587"
            }
        },
        "outcome": {
            "name": "Threatened",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Threatened"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "provenance": []
    },
    # 14. R -> xi        -> H
    {
        "rate_law": "1.0*Recognized*xi",
        "type": "NaturalConversion",
        "subject": {
            "name": "Recognized",
            "identifiers": {
                "ido": "0000511",
                "biomodels.species": "BIOMD0000000955:Recognized"
            },
            "context": {
                "property": "ncit:C25587"
            }
        },
        "outcome": {
            "name": "Healed",
            "identifiers": {
                "biomodels.species": "BIOMD0000000955:Healed"
            },
            "context": {
                "property": "ido:0000621"
            }
        },
        "provenance": []
    },
    # 15. T -> tau       -> E
    {
        "rate_law": "1.0*Threatened*tau",
        "type": "NaturalConversion",
        "subject": {
            "name": "Threatened",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Threatened"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "outcome": {
            "name": "Extinct",
            "identifiers": {
                "ncit": "C28554",
                "biomodels.species": "BIOMD0000000955:Extinct"
            },
            "context": {}
        },
        "provenance": []
    },
    # 16. T -> sigma     -> H
    {
        "rate_law": "1.0*Threatened*sigma",
        "type": "NaturalConversion",
        "subject": {
            "name": "Threatened",
            "identifiers": {
                "ido": "0000573",
                "biomodels.species": "BIOMD0000000955:Threatened"
            },
            "context": {
                "property": "ncit:C15220"
            }
        },
        "outcome": {
            "name": "Healed",
            "identifiers": {
                "biomodels.species": "BIOMD0000000955:Healed"
            },
            "context": {
                "property": "ido:0000621"
            }
        },
        "provenance": []
    }
]

# %%
# Thin-Thread Pipeline

model_mmt_templates = {'templates': model_mmt['templates']}
model_mmt_parameters = {'parameters': model_mmt['parameters']}

# Initial conditions
# Find all state variables
state_vars = [t['subject'] for t in model_mmt['templates'] if 'subject' in t.keys()]
state_vars.extend([t['outcome'] for t in model_mmt['templates'] if 'outcome' in t.keys()])
state_vars.extend([i for t in model_mmt['templates'] if 'controllers' in t.keys() for i in t['controllers']])
state_vars.extend([t['controller'] for t in model_mmt['templates'] if 'controller' in t.keys()])
state_vars_uniq = {hash(json.dumps(v, sort_keys = True, default = str, ensure_ascii = True).encode()): v for v in state_vars}
model_mmt_initials = {'initials': {var['name']: {**var, **{'value': None}} for var in state_vars_uniq.values()}}

# Populate with given values
for k, v in model_mmt['initials'].items():
    if k in model_mmt_initials['initials'].keys():
        model_mmt_initials['initials'][k]['value'] = v

# Get Petri net
res = requests.post(f'{REST_URL_MIRA}/to_petrinet', json = model_mmt)
if res.status_code == 200:
    model_petri = res.json()
else:
    model_petri = None

# Create artifact directory if not exist
source = 'demo'
info = 'BIOMD0000000955'
path = f'../../thin-thread-examples/{source}/{info}'
if os.path.exists(path) == False:
    os.mkdir(path)

# Write artifact files
for data, filename in zip([model_mmt, model_mmt_templates, model_mmt_parameters, model_mmt_initials, model_petri], ['model_mmt.json', 'model_mmt_templates.json', 'model_mmt_parameters.json', 'model_mmt_initials.json', 'model_petri.json']):

    if data != None:
        
        # SBML XML file
        if filename.split('.')[-1] == 'xml':

            # `src` directory
            if os.path.exists(path + '/src') == False:
                os.mkdir(path + '/src')
            
            # `src/main` directory
            if os.path.exists(path + '/src/main') == False:
                os.mkdir(path + '/src/main')
            
            with open(path + f'/src/main/{filename}', 'wb') as f:
                f.write(data)

        else:
            with open(path + f'/{filename}', 'w') as f:
                if isinstance(data, dict):
                    f.write(json.dumps(data, indent = 4))
                else:
                    f.write(data)

    else:
        print(f'Error: {info} {filename} data = None')

# %%
