
# Load packages
using MLStyle
using Catlab
using Decapodes
using Decapodes.decapodes
using Latexify


# Define the Halfar-Glen ice dynamics model
# as a SummationDecapode
model = @decapode begin

    h::Form0
    Γ::Form1
    n::Constant
    A::Constant
    ρ::Constant
    g::Constant

    ḣ == ∂ₜ(h)
    ḣ == ∘(⋆, d, ⋆)(Γ * d(h) * avg₀₁(mag(♯(d(h)))^(n-1)) * avg₀₁(h^(n+2)))
    Γ == (2/(n+2)) * A * (ρ * g)^n

end

# Same model with vector calculus operators
model_vec = @decapode begin
    h::Form0
    Γ::Form1
    n::Constant
    A::Constant
    ρ::Constant
    g::Constant

    ḣ == ∂ₜ(h)
    ḣ == div(Γ * grad(h) * mag(grad(h))^(n - 1) * h^(n + 2))
    Γ == (2/(n+2)) * A * (ρ * g)^n

end


# Convert model to DecaExpr and then pretty-print
println(sprint((io, x) -> Decapodes.pprint(io, x), Term(model)))
# Context:
#   h::Form0 over I
#   Γ::Form1 over I
#   n::Constant over I
#   A::Constant over I
#   ρ::Constant over I
#   g::Constant over I
#   ḣ::infer over I
#   •1::infer over I
#   mult_1::infer over I
#   •2::infer over I
#   •3::infer over I
#   •4::infer over I
#   •5::infer over I
#   •6::infer over I
#   •7::infer over I
#   •8::infer over I
#   1::Literal over I
#   •9::infer over I
#   •10::infer over I
#   2::Literal over I
#   sum_1::infer over I
#   mult_2::infer over I
#   mult_3::infer over I
#   mult_4::infer over I
#   •11::infer over I
#   sum_2::infer over I
#   •12::infer over I
#   •_8_1::infer over I
#   •_8_2::infer over I
# Equations:
# ḣ   = ∂ₜ(h)
# •2   = d(h)
# •7   = d(h)
# •6   = ♯(•7)
# •5   = mag(•6)
# •3   = avg₀₁(•4)
# •9   = avg₀₁(•10)
# •_8_1   = ⋆(mult_4)
# •_8_2   = d(•_8_1)
# ḣ   = ⋆(•_8_2)
# •8   = n - 1
# •4   = ^(•5, •8)
# •10   = ^(h, sum_1)
# mult_2   = Γ * •2
# mult_3   = mult_2 * •3
# mult_4   = mult_3 * •9
# •11   = 2 / sum_2
# •1   = ρ * g
# •12   = ^(•1, n)
# mult_1   = •11 * A
# Γ   = mult_1 * •12
# sum_1   = n + 2
# sum_2   = n + 2

# Save to text file
s = sprint((io, x) -> Decapodes.pprint(io, x), Term(model))
open("dome_model_de_flat_sm.txt", "w") do io
    println(io, s)
end    

# Substitute SSA equations
# ...
s_ = readlines("dome_model_de_flat_sm_.txt")

# Convert equations from Julia DSL to LaTeX
latexify(s_)


# Print same model in VEC notation
println(sprint((io, x) -> Decapodes.pprint(io, x), Term(model_vec)))
# Context:
#   h::Form0 over I
#   Γ::Form1 over I
#   n::Constant over I
#   A::Constant over I
#   ρ::Constant over I
#   g::Constant over I
#   ḣ::infer over I
#   •1::infer over I
#   mult_1::infer over I
#   •2::infer over I
#   •3::infer over I
#   •4::infer over I
#   •5::infer over I
#   •6::infer over I
#   1::Literal over I
#   •7::infer over I
#   2::Literal over I
#   sum_1::infer over I
#   mult_2::infer over I
#   mult_3::infer over I
#   mult_4::infer over I
#   •8::infer over I
#   sum_2::infer over I
#   •9::infer over I
# Equations:
# ḣ   = ∂ₜ(h)
# •2   = grad(h)
# •5   = grad(h)
# •4   = mag(•5)
# ḣ   = div(mult_4)
# •6   = n - 1
# •3   = ^(•4, •6)
# •7   = ^(h, sum_1)
# mult_2   = Γ * •2
# mult_3   = mult_2 * •3
# mult_4   = mult_3 * •7
# •8   = 2 / sum_2
# •1   = ρ * g
# •9   = ^(•1, n)
# mult_1   = •8 * A
# Γ   = mult_1 * •9
# sum_1   = n + 2
# sum_2   = n + 2

# Save to text file
s_vec = sprint((io, x) -> Decapodes.pprint(io, x), Term(model_vec))
open("dome_model_de_flat_sm_vec.txt", "w") do io
    println(io, s_vec)
end

s_vec_ = readlines("dome_model_de_flat_sm_vec_.txt")

# Convert equations from Julia DSL to LaTeX
latexify(s_vec_)

# Conversion from vector calculus operators to discrete calculus ones
vec_to_dec!(model_vec)
println(sprint((io, x) -> Decapodes.pprint(io, x), Term(model_vec)))
# Context:
#   h::Form0 over I
#   Γ::Form1 over I
#   n::Constant over I
#   A::Constant over I
#   ρ::Constant over I
#   g::Constant over I
#   ḣ::infer over I
#   •1::infer over I
#   mult_1::infer over I
#   •2::infer over I
#   •3::infer over I
#   •4::infer over I
#   •5::infer over I
#   •6::infer over I
#   1::Literal over I
#   •7::infer over I
#   2::Literal over I
#   sum_1::infer over I
#   mult_2::infer over I
#   mult_3::infer over I
#   mult_4::infer over I
#   •8::infer over I
#   sum_2::infer over I
#   •9::infer over I
#   •_5_1::infer over I
#   •_5_2::infer over I
# Equations:
# ḣ   = ∂ₜ(h)
# •2   = d(h)
# •5   = d(h)
# •4   = mag(•5)
# •_5_1   = ⋆(mult_4)
# •_5_2   = d(•_5_1)
# ḣ   = ⋆(•_5_2)
# •6   = n - 1
# •3   = ^(•4, •6)
# •7   = ^(h, sum_1)
# mult_2   = Γ * •2
# mult_3   = mult_2 * •3
# mult_4   = mult_3 * •7
# •8   = 2 / sum_2
# •1   = ρ * g
# •9   = ^(•1, n)
# mult_1   = •8 * A
# Γ   = mult_1 * •9
# sum_1   = n + 2
# sum_2   = n + 2
