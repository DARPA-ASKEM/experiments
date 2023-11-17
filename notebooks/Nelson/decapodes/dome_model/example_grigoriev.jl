# Halfar-Glen Ice Dome Example with Real DataFrames

# Replicating the notebook here: https://algebraicjulia.github.io/Decapodes.jl/dev/grigoriev/

# Load packages
import Decapodes as de
import CombinatorialSpaces as cs
import GeometryBasics as gb
import FileIO as fio
import NetCDF as netcdf
import Interpolations as interp
import LinearAlgebra as linalg
import ComponentArrays as comparr
# import OrdinaryDiffEq as ode
# import CairoMakie as cm


# Define alias
Point2D = gb.Point2{Float64}

# 1. Data Transformation Steps

# 1.1 Load the ice thickness distribution dataset for the Grigoriev ice cap
# From https://zenodo.org/api/records/7735970/files-archive
h_init_tif = fio.load("./grigoriev/inputs/LanderVT-Icethickness-Grigoriev-ice-cap-c32c9f7/Icethickness_Grigoriev_ice_cap_2021.tif")

# 1.2 Compute Cartesian coordinates associated with the dataset

# Taken from the bounding box UTM coordinates
const MIN_X = 243504.5
const MAX_X = 245599.8
const MIN_Y = 4648894.5
const MAX_Y = 4652179.7

# Note: 
# shape of h_init_tif is Y by X (not X by Y)
# can't just apply linalg.transpose on h_init_tif...
# must swap X and Y everywhere else
yx_coords = (
    range(MIN_Y, MAX_Y, length = size(h_init_tif, 1)), 
    range(MIN_X, MAX_X, length = size(h_init_tif, 2))
)

# 1.3 Create interpolation object over new uniform grid
h_init_interp = interp.LinearInterpolation(yx_coords, Float32.(h_init_tif))

# 1.4 Generate mesh for model discretization

# Specify resolution (~25 per pixel)
RES_Y = (MAX_Y - MIN_Y) / size(h_init_tif, 1)
RES_X = (MAX_X - MIN_X) / size(h_init_tif, 2)

# Generate mesh
include("../grid_meshes.jl")
s_primal = triangulated_grid(MAX_Y - MIN_Y, MAX_X - MIN_X, RES_Y, RES_X, Point3D)
s_primal[:point] = map(p -> p + Point3D(MIN_Y, MIN_X, 0), s_primal[:point])

# Generate dual mesh
s_dual = cs.EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s_primal)
cs.subdivide_duals!(s_dual, cs.Barycenter())

# Save meshes
fio.save("./grigoriev/primal_mesh.obj", gb.Mesh(s_dual, primal = true))
fio.save("./grigoriev/dual_mesh.obj", gb.Mesh(s_dual, primal = false))

# 1.5 Parameterization

# Constants
n = 3
ρ = 910
g = 9.8101
A = fill(1e-16, cs.ne(s_dual))

# Initial condition
# s_dual???
h_init = map(s_dual[:point]) do (y, x, _)
    tif_val = h_init_interp(y, x)
    tif_val < 0.0 ? 0.0 : tif_val
end

# Save as NetCDF
f = joinpath("grigoriev/inputs/h_init.nc")
isfile(f) && rm(f)
netcdf.nccreate(
    f, "h_init", 
    "vertex_index", [i for i in eachindex(s_dual[:point])], Dict("name" => "mesh-vertex index", "units" => "None"),
    atts = Dict("name" => "ice height at time = 0, Grigoriev ice cap", "units" => "m")
)
netcdf.ncwrite(h_init, f, "h_init")
netcdf.ncinfo(f)

# u_init = comparr.ComponentArray(h = h_init, stress_A = A)
# constants_parameters = (
#     n = n,
#     stress_ρ = ρ, 
#     stress_g = g,
#     stress_A = A
# )
u_init = comparr.ComponentArray(h = h_init, A = A)
constants_parameters = (
    n = n,
    ρ = ρ, 
    g = g,
    A = A
)
t_init = 0
t_end = 1e1

# 1.6 Import Previously Created Model

# ???
# dome_model = de.read_json_acset(de.SummationDecapode{Symbol, Symbol, Symbol}, "./dome_model_sm.json")

dome_model = @de.decapode begin
    h::Form0
    Γ::Form1
    (A,ρ,g,n)::Constant

    ḣ == ∂ₜ(h)
    ḣ == ∘(⋆, d, ⋆)(Γ * d(h) * avg₀₁(mag(♯(d(h)))^(n-1)) * avg₀₁(h^(n+2)))
    Γ == (2/(n+2))*A*(ρ*g)^n
end

# 1.7 Helper Functions

include("../sharp_op.jl")

function generate(sd, my_symbol; hodge=cs.GeometricHodge())
    ♯_m = ♯_mat(sd)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for e in 1:ne(s)
        append!(J, [s[e,:∂v0],s[e,:∂v1]])
        append!(I, [e,e])
        append!(V, [0.5, 0.5])
    end
    avg_mat = sparse(I,J,V)
    op = @de.match my_symbol begin
      :♯ => x -> begin
        ♯(sd, EForm(x))
      end
      :mag => x -> begin
        norm.(x)
      end
      :avg₀₁ => x -> begin
        avg_mat * x
      end
      :^ => (x,y) -> x .^ y
      :* => (x,y) -> x .* y
      :abs => x -> abs.(x)
      :show => x -> begin
        println(x)
        x
      end
      x => error("Unmatched operator $my_symbol")
    end
    return (args...) -> op(args...)
end

# 1.8 Simulation

# Generate simulation
sim = eval(de.gensim(dome_model, dimension = 2))
fm = sim(s_dual, generate)

# # Run
# @info("Solving Grigoriev Ice Cap")
# prob = ODEProblem(fm, u_init, (t_init, t_end), constants_and_parameters)
# soln = solve(prob, Tsit5())
# @show soln.retcode
# @info("Done")
# # @save "grigoriev.jld2" soln

# # Extract variable from solution object
# u = Array{Float64, 2}(undef, length(soln.t), length(soln.u[1]))
# for i in eachindex(soln.t)
#     for j in eachindex(soln.u[1])
#         u[i, j] = soln.u[i][j]
#     end
# end

# # Save output
# f = joinpath("grigoriev/outputs/outputs.nc")
# isfile(f) && rm(f)
# netcdf.nccreate(
#     f, "h", 
#     "vertex_index", [i for i in eachindex(s_dual[:point])], Dict("name" => "mesh-vertex index", "units" => "None"),
#     atts = Dict("name" => "ice height, Grigoriev ice cap", "units" => "m")
# )
# netcdf.ncwrite(u, f, "h")
# netcdf.ncinfo(f)
