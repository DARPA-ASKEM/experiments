# See Genine framework example here
# https://genieframework.github.io/Genie.jl/dev/tutorials/5--Handling_Query_Params.html
#
# See also this notebook
# https://github.com/AlgebraicJulia/Structured-Epidemic-Modeling/
#
#
using UUIDs
using Genie
using Genie.Requests
using Genie.Renderer.Json

using Catlab
using Catlab.CategoricalAlgebra 
using Catlab.Programs
using Catlab.WiringDiagrams
using Catlab.Graphics.Graphviz
using AlgebraicPetri


# mutable struct MyStruct
#   x::Float64
#   y::Float64
#   s::String
# end
# 
# infectious_type = LabelledPetriNet([:Pop],
#   :interact=>((:Pop, :Pop)=>(:Pop, :Pop)), 
#   :t_disease=>(:Pop=>:Pop),
#   :t_strata=>(:Pop=>:Pop)
# )
# 
# s, = parts(infectious_type, :S)
# t_interact, t_disease, t_strata = parts(infectious_type, :T)
# i_interact1, i_interact2, i_disease, i_strata = parts(infectious_type, :I)
# o_interact1, o_interact2, o_disease, o_strata = parts(infectious_type, :O);
# 
# infectious_type = map(infectious_type, Name=name->nothing); # remove names to allow for the loose ACSet transform to be natural
# 
# 
# SIR = LabelledPetriNet([:S, :I, :R],
#   :inf => ((:S, :I)=>(:I, :I)),
#   :rec => (:I=>:R),
#   :id => (:S => :S),
#   :id => (:I => :I),
#   :id => (:R => :R)
# )
# 
# typed_SIR = ACSetTransformation(SIR, infectious_type,
#   S = [s, s, s],
#   T = [t_interact, t_disease, t_strata, t_strata, t_strata],
#   I = [i_interact1, i_interact2, i_disease, i_strata, i_strata, i_strata],
#   O = [o_interact1, o_interact2, o_disease, o_strata, o_strata, o_strata],
#   Name = name -> nothing # specify the mapping for the loose ACSet transform
# );
# 
# # Grabs the "foobar" query-parameter, if available, otherwise default
# route("/test-param") do
#   xyz = MyStruct(123.0, 456.0, "")
#   xyz.s = getpayload(:foobar, "eek")
#   xyz |> json
# end
# 
# route("/test-petri-raw") do
#     infectious_type |> json
# end
# 
# route("/test-petri-typed") do
#     typed_SIR |> json
# end
# 
# dict = Dict()
# 
# # Test simple POST request that accumulates into a dictionary
# route("/test-post", method = POST) do
#     xyz = jsonpayload()
#     for (key, value) in xyz
#         dict[key] = value
#         print(key); print(value)
#     end
#     json(dict)
# end


"""
Start
"""
const modelDict = Dict{String, LabelledPetriNet}()

# Retrieve a model
route("/api/models/:model_id") do
    key = payload(:model_id)
    println(" Checking key $(key) => $(haskey(modelDict, key))")

    if !haskey(modelDict, key) 
        return json("not found")
    end
    model = modelDict[key]
    return json(model)
end


# Create a new empty model
route("/api/models", method = PUT) do
    # @info "Creating new model"
    modelId = string(UUIDs.uuid4())

    model = LabelledPetriNet()
    modelDict[modelId] = model

    println(modelDict)

    return json(
         Dict([ 
               (:id, modelId)
         ])
    )
end


# Add nodes and edges, a more natural way of adding components instead of solely relying 
# on indices
route("/api/models/:model_id", method = POST) do
    key = payload(:model_id)
    println(" Checking key $(key) => $(haskey(modelDict, key))")

    if !haskey(modelDict, key) 
        return json("not found")
    end

    model = modelDict[key]
    data = jsonpayload()

    # nodes, need to be processed first, otherwise index assignment will fail for edges
    if haskey(data, "nodes")
        for n in data["nodes"]
            if n["type"] == "S"
                add_parts!(model, :S, 1, sname=Symbol(n["name"]))
            elseif n["type"] == "T"
                add_parts!(model, :T, 1, tname=Symbol(n["name"]))
            end
        end
    end

    # edges
    if haskey(data, "edges")
        attrs = model.attrs

        for e in data["edges"]

            source = Symbol(e["source"])
            target = Symbol(e["target"])

            if isnothing(findfirst(x -> x == source, attrs.sname)) == false 
                sourceIdx = findfirst(x -> x == source, attrs.sname)
                targetIdx = findfirst(x -> x == target, attrs.tname)

                add_parts!(model, :I, 1, is=sourceIdx, it=targetIdx)
            end

            if isnothing(findfirst(x -> x == source, attrs.tname)) == false
                sourceIdx = findfirst(x -> x == source, attrs.tname)
                targetIdx = findfirst(x -> x == target, attrs.sname)

                add_parts!(model, :O, 1, os=sourceIdx, ot=targetIdx) 
            end
        end
    end
    
    # Serialize back
    # println("Serializing back")
    modelDict[key] = model

    return json("done")
end


# Get JSON representation of the model
route("/api/models/:model_id/json") do
    key = payload(:model_id)

    if !haskey(modelDict, key) 
        return json("not found")
    end

    model = modelDict[key]
    dataOut = generate_json_acset(model)

    return dataOut
end


# Start the web-app
up(8888, async = false)
