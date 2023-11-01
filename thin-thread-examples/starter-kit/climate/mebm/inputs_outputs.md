# Moist Energy Balance Model

## Notebook

Example inputs and outputs generated using `example_notebook.py`

## Inputs

* Initial temperature (`init_temp_type = "legendre", low = 250, high = 300`)
* Insolation type (`insol_type = "perturbation", perturb_center = 15, perturb_spread = 4.94, perturb_intensity = 10`)
* Albedo (`al_feedback = True, alb_ice = 0.6, alb_water = 0.2`)
* Outgoing longwave radiation (`olr_type = "full_radiation", A = None, B = None, emissivity = None`)

## Outputs

* Energy flux equator position (`itcz.log`)
* State variable values of control experiment (`ctrl.npz`)
* State variables values of scenario (`simulation_data.npz`)
* Several plots of the state variables and derived quantities (`mse.png, radiation.png, temp.png, dmse.png, dtemp.png, differences_transports.png`)
