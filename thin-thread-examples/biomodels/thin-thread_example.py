import datetime
import requests
import json


url = "http://localhost:8001/"

#### Person ####
def create_person(url=url):
    path = "persons"

    payload = json.dumps(
        {
            "name": "Adam Smith",
            "email": "Adam@test.io",
            "org": "Uncharted",
            "website": "",
            "is_registered":True
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)
    return response.json()


#### Project ####

def create_project(url=url):
    path = "projects"

    payload = json.dumps(
        {
            "name": "My Project",
            "description": "First project in TDS",
            "assets":{},
            "status": "active",
        }
    )
    headers = {"Content-Type": "application/json"}

    # return project id (p1)
    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)
    return response.json()



#### Framework ####

def create_framework(url=url):
    path="frameworks"

    payload = json.dumps(
        {
            "name": "Petri Net",
            "version": "0.0.1",
            "semantics": "semantics_go_here",
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)
    return response.text



#### Publication ####
# path = "publications"

# payload = json.dumps(
#     {
#         "xdd_uri": "616cf16267467f7269ccde6f",
#         "doi": "10.1038/s41591-020-0883-7"
#     }
# )
# headers = {"Content-Type": "application/json"}

# # return resource_id (a1)
# response = requests.request("POST", url + path, headers=headers, data=payload)

# print(response.text)



#### Asset ####
# path = "assets"

# payload = json.dumps(
#     {
#         "resource_id": "a1",
#         "project_id": "p1"
#     }
# )
# headers = {"Content-Type": "application/json"}

# response = requests.request("POST", url + path, headers=headers, data=payload)

# print(response.text)



#### Intermediate ####
# MMT
# path = "intermediates"

# # load model contents 
# with open("../BioModels/model_mmt-templates.json", "r") as f:
#     intermediate_content = json.load(f)


# payload = json.dumps(
#     {
#         "type": "bilayer",
#         "source": "mrepresentationa",
#         "content": json.dumps(intermediate_content),
#     }
# )
# headers = {"Content-Type": "application/json"}

# response = requests.request("POST", url + path, headers=headers, data=payload)

# print(response.text)




#### Model ####
# includes metadata, parameters, and content


def create_model(url=url):
    path = "models"

    # load model contents 
    with open("first_model/model_petri.json", "r") as f:
        model_content = json.load(f)

    # load parameters of the model and set the type values
    parameter_types = {}
    with open("first_model/model_mmt-parameters.json", "r") as f:
        parameters = json.load(f)
        for parameter_name, parameter_value in parameters.get("parameters").items():
            parameter_types[parameter_name] = str(type(parameter_value).__name__)

    payload = json.dumps(
        {
            "name": "Covid SIDARHTE",
            "description": "Petri net model to predict covid movement patters in a community",
            "content": json.dumps(model_content),
            "framework": "Petri Net",
            "parameters": parameter_types,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)
    return response.json()


#### Associations ####

def create_association(person_id, resource_id, resource_type, url=url):
    path="associations"

    payload = json.dumps(
        {
        "person_id": person_id,
        "resource_id": resource_id,
        "resource_type": resource_type,
        "role": "author"
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)
    return response.text


#### Simulation plan ####


def create_simulation_plan(model_id):
    path = "simulations/plans"

    # load simulation plan contents as json
    with open("first_model/simulation-plan_ATE.json", "r") as f:
        simulation_body = json.load(f)

    parameter_types={"iterations":("5", "int")}
    
    # with open("first_model/model_mmt-parameters.json", "r") as f:
    #         parameters = json.load(f)
    #         for parameter_name, parameter_value in parameters.get("parameters").items():
    #             parameter_types[parameter_name] = (str(parameter_value), str(type(parameter_value).__name__))

    payload = json.dumps(
        {
            "name": "simulation_1",
            "model_id":model_id,
            "description": "description",
            "simulator": "",
            "query": "My query",
            "content": json.dumps(simulation_body),
            "parameters": parameter_types,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url + path, headers=headers, data=payload)

    print(response.text)



person=create_person()
print(person)
person_id=person.get('person_id')
project=create_project()
project_id = project.get("project_id")
create_framework()


model=create_model()
model_id=model.get('model_id')
print('her')
print(model_id, person_id)
create_association(person_id=int(person_id), resource_id=int(model_id), resource_type="model")

simulation_id=create_simulation_plan(model_id=model_id)
