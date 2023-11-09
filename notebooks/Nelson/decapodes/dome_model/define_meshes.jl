# 

# Minimal packages
import Decapodes as de
import CombinatorialSpaces as cs
import LinearAlgebra as la
import GeometryBasics as gb
import GeometryBasics: Point2, Point3
import FileIO as fio

# Define alias
Point2D = Point2{Float64}
Point3D = Point3{Float64}

# Define a 2D rectangular triangulated grid using a helper function
function triangulated_grid(max_x, max_y, dx, dy, point_type)

    s = cs.EmbeddedDeltaSet2D{Bool, point_type}()
  
    # Place equally-spaced points in a max_x by max_y rectangle.
    coords = point_type == Point3D ? map(x -> point_type(x..., 0), Iterators.product(0:dx:max_x, 0:dy:max_y)) : map(x -> point_type(x...), Iterators.product(0:dx:max_x, 0:dy:max_y))
    # Perturb every other row right by half a dx.
    coords[:, 2:2:end] = map(coords[:, 2:2:end]) do row
      if point_type == Point3D
        row .+ [dx/2, 0, 0]
      else
        row .+ [dx/2, 0]
      end
    end
    # The perturbation moved the right-most points past max_x, so compress along x.
    map!(coords, coords) do coord
      if point_type == Point3D
        la.diagm([max_x/(max_x+dx/2), 1, 1]) * coord
      else
        la.diagm([max_x/(max_x+dx/2), 1]) * coord
      end
    end
  
    cs.add_vertices!(s, length(coords), point = vec(coords))
  
    nx = length(0:dx:max_x)
  
    # Matrix that stores indices of points.
    idcs = reshape(eachindex(coords), size(coords))
    # Only grab vertices that will be the bottom-left corner of a subdivided square.
    idcs = idcs[begin:end-1, begin:end-1]
    
    # Subdivide every other row along the opposite diagonal.
    for i in idcs[:, begin+1:2:end]
      cs.glue_sorted_triangle!(s, i, i+nx, i+nx+1)
      cs.glue_sorted_triangle!(s, i, i+1, i+nx+1)
    end
    for i in idcs[:, begin:2:end]
      cs.glue_sorted_triangle!(s, i, i+1, i+nx)
      cs.glue_sorted_triangle!(s, i+1, i+nx, i+nx+1)
    end
  
    # Orient and return.
    s[:edge_orientation] = true
    cs.orient!(s)
    s
end

# 2D rectangle in 2D space
s_prime_2D_rect = triangulated_grid(10_000, 10_000, 800, 800, Point3D)
s_2D_rect = cs.EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s_prime_2D_rect)
cs.subdivide_duals!(s_2D_rect, cs.Barycenter())
fio.save("./2D_rect/primal_mesh.obj", gb.Mesh(s_2D_sph, primal = true))
fio.save("./2D_rect/dual_mesh.obj", gb.Mesh(s_2D_sph, primal = false))

# 2D icosphere in 3D space
s_prime_2D_sph = de.loadmesh(de.Icosphere(3, 10_000))
s_2D_sph = cs.EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s_prime_2D_sph)
cs.subdivide_duals!(s_2D_sph, cs.Barycenter())
fio.save("./2D_sph/primal_mesh.obj", gb.Mesh(s_2D_sph, primal = true))
fio.save("./2D_sph/dual_mesh.obj", gb.Mesh(s_2D_sph, primal = false))

# 2D teapot in 3D space
# download("https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj", "teapot.obj")
s_prime_2D_tea = cs.EmbeddedDeltaSet2D("./2D_tea/teapot.obj")
s_2D_tea = cs.EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s_prime_2D_tea)
cs.subdivide_duals!(s_2D_tea, cs.Circumcenter())
fio.save("./2D_tea/primal_mesh.obj", gb.Mesh(s_2D_tea, primal = true))
fio.save("./2D_tea/dual_mesh.obj", gb.Mesh(s_2D_tea, primal = false))
