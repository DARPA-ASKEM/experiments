import sys
import os
import json

# Request
f = open("request.json", "r")
request_json = json.loads(f.read())

# Funman imports
from funman import Funman
from funman.config import FUNMANConfig
from funman.model.model import _wrap_with_internal_model
from funman.scenario.scenario import AnalysisScenario
from funman.model.petrinet import PetrinetModel
from funman.representation.parameter_space import ParameterSpace
from funman.server.query import (
    FunmanProgress,
    FunmanResults,
    FunmanWorkUnit,
)
from pydantic import TypeAdapter
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet

adapter = TypeAdapter(GeneratedPetrinet)

def run_validate(model: PetrinetModel, request):
    current_results = FunmanResults(
        id="test-fun",
        model=model,
        request=request,
        parameter_space=ParameterSpace(),
    )

    # Invoke solver
    work = FunmanWorkUnit(id="test_fun", model=model, request=request)
    f = Funman()
    scenario = work.to_scenario()
    config = (
        FUNMANConfig()
        if work.request.config is None
        else work.request.config
    )
    result = f.solve(
        scenario,
        config=config,
        # haltEvent=self._halt_event,
    )
    print("Done solver portion", file = sys.stderr)
    current_results.finalize_result(scenario, result)
    print(current_results.model_dump_json(by_alias=False))
    return current_results.model_dump_json(by_alias=False)


model = adapter.validate_python(request_json["model"])
model = _wrap_with_internal_model(model)
request = request_json["request"]
result = run_validate(model, request)
