## Catlab-service
Testing a simple Julia-based webserver wrapping Catalb.

### Dependencies
Julia 1.7.3

### TODO
- composition
- typing
- stratification

### Running
```
julia main.jl
```

### Testing
```
# Create a model
curl -XPUT localhost:8888/api/models

# Add node
curl -XPOST -H "Content-type: application/json" localhost:8888/api/models/:id -d'
{
  "nodes": [
    { "name": "Hello", "type": "S" },
    { "name": "World", "type": "T" }
  ]
}'

# Add edge
curl -XPOST -H "Content-type: application/json" localhost:8888/api/models/:id -d'
{
  "edges": [
    { "source": "Hello", "target": "World" }
  ]
}'

# Add edge
curl -XPOST -H "Content-type: application/json" localhost:8888/api/models/:id -d'
{
  "edges": [
    { "source": "World", "target": "Hello" }
  ]
}'

# Get model
curl -XGET -H "Content-type: application/json" localhost:8888/api/models/:id

# Get JSON representation of model
curl -XGET -H "Content-type: application/json" localhost:8888/api/models/:id/json 
```
