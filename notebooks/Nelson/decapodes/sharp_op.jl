# TODO: Upstream these operators to CombinatorialSpaces.jl:
# https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/pull/59/files

using SparseArrays
using StaticArrays

#""" Divided weighted normals by | σⁿ | .
#This weighting is that used in equation 5.8.1 from Hirani.
#See Hirani §5.8.
#"""
#♯_denominator(s::cs.AbstractDeltaDualComplex2D, _::Int, t::Int) =
#  volume(2,s,t)

""" Divided weighted normals by | ⋆v | .
This weighting is NOT that of equation 5.8.1, but a different weighting scheme.
We essentially replace the denominator in equation 5.8.1 with | ⋆v | . This
may be what Hirani intended, and perhaps the denominator | σⁿ | in that equation
is either a mistake or clerical error.
See Hirani §5.8.
"""
♯_denominator(s::cs.AbstractDeltaDualComplex2D, v::Int, _::Int) =
  sum(dual_volume(2,s, elementary_duals(0,s,v)))

""" Find a vector orthogonal to e pointing into the triangle shared with v.
"""
function get_orthogonal_vector(s::cs.AbstractDeltaDualComplex2D, v::Int, e::Int)
  e_vec = cs.point(s, cs.tgt(s, e)) - cs.point(s, cs.src(s, e))
  e_vec /= cs.norm(e_vec)
  e2_vec = cs.point(s, v) - cs.point(s, cs.src(s, e))
  e2_vec - dot(e2_vec, e_vec)*e_vec
end

function ♯_assign!(♯_mat::AbstractSparseMatrix, s::cs.AbstractDeltaDualComplex2D, 
  v₀::Int, _::Int, t::Int, i::Int, tri_edges::SVector{3, Int}, tri_center::Int,
  out_vec)
  for e in deleteat(tri_edges, i)
    v, sgn = cs.src(s,e) == v₀ ? (cs.tgt(s,e), -1) : (cs.src(s,e), +1)
    # | ⋆vₓ ∩ σⁿ |
    dual_area = sum(dual_volume(2,s,d) for d in elementary_duals(0,s,v)
                    if s[s[d, :D_∂e0], :D_∂v0] == tri_center)
    area = ♯_denominator(s, v, t)
    ♯_mat[v,e] += sgn * sign(1,s,e) * (dual_area / area) * out_vec
  end
end

function ♯_mat(s::cs.AbstractDeltaDualComplex2D)
  #♯_mat = spzeros(attrtype_type(s, :Point), (cs.nv(s), cs.ne(s)))
  ♯_mat = spzeros(Point3D, (cs.nv(s), cs.ne(s)))
  for t in cs.triangles(s)
    tri_center, tri_edges = cs.triangle_center(s,t), cs.triangle_edges(s,t)
    for (i, (v₀, e₀)) in enumerate(zip(cs.triangle_vertices(s,t), tri_edges))
      out_vec = get_orthogonal_vector(s, v₀, e₀)
      h = cs.norm(out_vec)
      #out_vec /= DS == DesbrunSharp() ? h : h^2
      out_vec /= h^2
      ♯_assign!(♯_mat, s, v₀, e₀, t, i, tri_edges, tri_center, out_vec)
    end
  end
  ♯_mat
end
