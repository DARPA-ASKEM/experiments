begin
    #= none:3 =#
    function simulate(mesh, operators, hodge = cs.GeometricHodge())
        #= none:3 =#
        #= none:6 =#
        begin
            #= none:8 =#
            (M_d₀, d₀) = de.default_dec_matrix_generate(mesh, :d₀, hodge)
            #= none:9 =#
            (var"M_⋆₁", ⋆₁) = de.default_dec_matrix_generate(mesh, :⋆₁, hodge)
            #= none:10 =#
            (M_dual_d₁, dual_d₁) = de.default_dec_matrix_generate(mesh, :dual_d₁, hodge)
            #= none:11 =#
            (var"M_⋆₀⁻¹", ⋆₀⁻¹) = de.default_dec_matrix_generate(mesh, :⋆₀⁻¹, hodge)
            #= none:12 =#
            ♯ = operators(mesh, :♯)
            #= none:13 =#
            mag = operators(mesh, :mag)
            #= none:14 =#
            avg₀₁ = operators(mesh, :avg₀₁)
            #= none:15 =#
            (^) = operators(mesh, :^)
        end
        #= none:18 =#
        begin
            #= none:20 =#
            var"•2" = Vector{Float64}(undef, nparts(mesh, :E))
            #= none:21 =#
            var"•7" = Vector{Float64}(undef, nparts(mesh, :E))
            #= none:22 =#
            var"•_8_1" = Vector{Float64}(undef, nparts(mesh, :E))
            #= none:23 =#
            var"•_8_2" = Vector{Float64}(undef, nparts(mesh, :V))
            #= none:24 =#
            ḣ = Vector{Float64}(undef, nparts(mesh, :V))
        end
        #= none:27 =#
        f(du, u, p, t) = begin
                #= none:27 =#
                #= none:30 =#
                begin
                    #= none:32 =#
                    h = (findnode(u, :h)).values
                    #= none:33 =#
                    A = p.A
                    #= none:34 =#
                    ρ = p.ρ
                    #= none:35 =#
                    g = p.g
                    #= none:36 =#
                    n = p.n
                    #= none:37 =#
                    var"1" = 1.0
                    #= none:38 =#
                    var"2" = 2.0
                end
                #= none:41 =#
                mul!(var"•2", M_d₀, h)
                #= none:42 =#
                mul!(var"•7", M_d₀, h)
                #= none:43 =#
                var"•6" = ♯(var"•7")
                #= none:44 =#
                var"•5" = mag(var"•6")
                #= none:45 =#
                var"•8" = n .- var"1"
                #= none:46 =#
                var"•4" = var"•5" ^ var"•8"
                #= none:47 =#
                var"•1" = ρ .* g
                #= none:48 =#
                var"•12" = var"•1" ^ n
                #= none:49 =#
                sum_1 = (.+)(n, var"2")
                #= none:50 =#
                sum_2 = (.+)(n, var"2")
                #= none:51 =#
                var"•3" = avg₀₁(var"•4")
                #= none:52 =#
                var"•10" = h ^ sum_1
                #= none:53 =#
                var"•11" = var"2" / sum_2
                #= none:54 =#
                mult_1 = var"•11" .* A
                #= none:55 =#
                Γ = mult_1 .* var"•12"
                #= none:56 =#
                var"•9" = avg₀₁(var"•10")
                #= none:57 =#
                mult_2 = Γ .* var"•2"
                #= none:58 =#
                mult_3 = mult_2 .* var"•3"
                #= none:59 =#
                mult_4 = mult_3 .* var"•9"
                #= none:60 =#
                mul!(var"•_8_1", var"M_⋆₁", mult_4)
                #= none:61 =#
                mul!(var"•_8_2", M_dual_d₁, var"•_8_1")
                #= none:62 =#
                mul!(ḣ, var"M_⋆₀⁻¹", var"•_8_2")
                #= none:64 =#
                (findnode(du, :h)).values .= ḣ
            end
    end
end