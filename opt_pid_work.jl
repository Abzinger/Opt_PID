module Opt_PID_work

#
# The main difference between this and the InfDecomp module is:
#  * here, we maximize H(X|YZ) instead of minimizing -H(X|YZ)
#

using Opt_PID_data: My_Eval, condEntropy, ∇f
# using MathProgBase:
#     initialize, features_available,
#     isobjlinear, isobjquadratic, isconstrlinear,
#     eval_f, eval_g, eval_grad_f,
#     jac_structure, eval_jac_g, eval_jac_prod, eval_jac_prod_t,
#     hesslag_structure, eval_hesslag #, eval_hesslag_prod
using MathProgBase

function MathProgBase.initialize(e::My_Eval, requested_features::Vector{Symbol})
    print("initialize")
    for feat in requested_features
        if feat ∉ [:Grad,:Jac,:JacVec,:Hess]
            error("Opt_PID_work.initialize():\n-   JuliaOpt:MathProgBase is asking for a feature ($feat) that I don't have.-   Maybe use another solver?")
        end
    end
    print("END initialize")
    ;
end


# features_available()
MathProgBase.features_available(::My_Eval) = features_list

# Properties:
MathProgBase.isobjlinear(::My_Eval)           = false
MathProgBase.isobjquadratic(::My_Eval)        = false
MathProgBase.isconstrlinear(::My_Eval, ::Integer) = true

#
# ------------------------------------------
# E v a l u a t i o n :   0 t h   o r d e r
# ------------------------------------------
function MathProgBase.eval_f{TFloat_2,TFloat}(e::My_Eval, x::Vector{TFloat_2},dummy::TFloat=Float64(0)) :: TFloat
    print("eval_f")
    local condent::Float64
    if e.TmpFloat==BigFloat
        prc = precision(BigFloat)
        setprecision(BigFloat,e.bigfloat_nbits)
        condent = condEntropy(e,x,BigFloat(0.))
        setprecision(BigFloat,prc)
    else
        condent = condEntropy(e,x,Float64(0))
    end
    print("END eval_f")
    return condent
    ;
end #^ eval_f()

# eval_g --- eval of constraint into g
function MathProgBase.eval_g(e::My_Eval, g::Vector{Float64}, x::Vector{Float64})  :: Void
    print("eval_g")
    g .= reshape( reshape(x,1,e.n)*e.Gt , e.m, ) .- e.rhs
    return nothing
    ;
end # eval_g()

# eval_grad_f --- eval gradient of objective function
function MathProgBase.eval_grad_f(e::My_Eval, g::Vector{Float64}, x::Vector{Float64}) :: Void
    print("eval_grad_f")
    if e.TmpFloat==BigFloat
        prc = precision(BigFloat)
        setprecision(BigFloat,e.bigfloat_nbits)
        ∇f(e,g,x,BigFloat(0.))
        setprecision(BigFloat,prc)
    else
        ∇f(e,g,x,Float64(0))
    end
    print("END eval_grad_f")
    ;
end # eval_grad_f()


# Constraint Jacobian
# jac_structure() --- zero-nonzero pattern of constraint Jacobian
MathProgBase.jac_structure(e::My_Eval) :: Tuple{Vector{Int64},Vector{Int64}}   =  ( e.Gt_L , e.Gt_K )
# Note: Gt is transposed, so K and L are swapped [K: rows of G^T; L: columns of G^T]

# eval_jac_g() --- constraint Jacobian   -> J
function MathProgBase.eval_jac_g(e::My_Eval, J::Vector{Float64}, x::Vector{Float64}) :: Void
    J .= e.Gt.nzval
    ;
end # eval_jac_g()


# eval_jac_prod() --- constraint_Jacobian * w   -> y
function MathProgBase.eval_jac_prod(e::My_Eval, y::Vector{Float64}, x::Vector{Float64}, w::Vector{Float64}) :: Void
    y .= reshape( reshape(w,1,e.n)*e.Gt , e.m, )
    ;
end # eval_jac_prod()


# eval_jac_prod_t() --- constraint_Jacobian^T * w  -> y
function MathProgBase.eval_jac_prod_t{T<:AbstractFloat}(e::My_Eval, y::Vector{T}, x::Vector{T}, w::Vector{T}) :: Void
    y .= e.Gt*w
    ;
end


# ------------------------------------------
# E v a l u a t i o n :   2 n d   o r d e r
# ------------------------------------------

# Lagrangian:
# L(x, (σ,μ) ) = σ f(x) + μ' G(x)
# Since G(.) is linear, it's Hessian is 0 anyway, so it won't bother us.

# hesslag_structure() --- zero-nonzero pattern of Hessian [wrt x] of the Lagrangian
function MathProgBase.hesslag_structure(e::My_Eval)  :: Tuple{Vector{Int64},Vector{Int64}}
    K = Vector{Int64}()
    L = Vector{Int64}()
    counter = 0
    for y = 1:e.n_y
        for z = 1:e.n_z
            # Start with the diagonal
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    counter += 1
                    push!(K,i)
                    push!(L,i)
                end
            end
            # Now off-diagonal.
            # H is  treated as a symmetric matrix, but:
            # if both (k,l) & (l,k) are present, their values will be added!
            for x = 1:e.n_x
                for u = 1:(x-1)
                    i_x = e.varidx[x,y,z]
                    i_u = e.varidx[u,y,z]
                    if i_x>0 && i_u>0
                        counter += 1
                        push!(K,i_x)
                        push!(L,i_u)
                    end
                end
            end
        end# for z
    end #^ for y
    return (K,L)
    ;
end # hesslag_structure()


function Hess{TFloat}(e::My_Eval, H::Vector{Float64}, p::Vector{Float64}, σ::Float64, dummy::TFloat) :: Void
    counter = 0
    for y = 1:e.n_y
        for z = 1:e.n_z
            # make marginal P(*yz)
            P_yz  ::TFloat = TFloat(0.)
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    P_yz += p[i]
                end
            end

            # now: for all pairs x,u we have:
            # if x ≠  u:   -1/P_yz;
            # if x == u:   ( P_yz - P(xyz) )/(  P_yz * P(xyz) )

            # Start with the diagonal
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    counter += 1
                    P_xyz = p[i]
                    H[counter] = Float64(   (P_xyz == 0 ) ?  1.e50  : TFloat(σ)*( P_yz - P_xyz )/(  P_yz * P_xyz )  )
                end
            end
            # Now off-diagonal.
            # H is  treated as a symmetric matrix, but:
            # if both (k,l) & (l,k) are present, their values will be added!
            for x = 1:e.n_x
                for u = 1:(x-1)
                    i_x = e.varidx[x,y,z]
                    i_u = e.varidx[u,y,z]
                    if i_x>0 && i_u>0
                        counter += 1
                        H[counter] = -TFloat(σ)/P_yz
                    end
                end
            end
        end#for z
    end# for y
    ;
end # eval_hesslag()



# eval_hesslag() --- Hessian [wrt x] of the Lagrangian
function MathProgBase.eval_hesslag(e::My_Eval, H::Vector{Float64}, x::Vector{Float64}, σ::Float64, μ::Vector{Float64}) :: Void
    if e.TmpFloat==BigFloat
        prc = precision(BigFloat)
        setprecision(BigFloat,e.bigfloat_nbits)
        Hess(e,H,x,σ,BigFloat(0.))
        setprecision(BigFloat,prc)
    else
        Hess(e,H,x,σ,Float64(0))
    end
    ;
end # eval_hesslag()


# eval_hesslag() --- ( Hessian [wrt x] of the Lagrangian ) * v
# function eval_hesslag_prod{T<:AbstractFloat}(e::My_Eval, h::Vector{T}, x::Vector{T}, v::Vector{T}, σ::T, μ::Vector{T}) :: Void
#     h .= σ .* Hf(x)*v
#     return nothing
# end

end #module Opt_PID_work
