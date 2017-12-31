include("opt_pid_data.jl")
include("opt_pid_work.jl")
module Opt_PID


using Opt_PID_data: My_Eval, create_My_Eval, set_rhs!,  condEntropy
using Mosek
using MathProgBase

struct PID_Results
    support  :: Array{Tuple{Int,Int,Int}}    # support of the following two vectors
    q        :: Vector{Float64}              # argmax H(X|YZ)
    ∇        :: Vector{Float64}              # (super-)gradient of H(X|YZ) in p
    p        :: Vector{Float64}              # current iterate (3-dim array!)
end

struct PID_Data
    bits     :: Float64
    myeval   :: My_Eval

    results  :: PID_Results

    # Storage --- this things are only here because I don't want to alloc mem for them in every iteration
    _space   :: Array{Float64,3} # 3D-storage for p
    _lb      :: Vector{Float64}
    _ub      :: Vector{Float64}
    _l       :: Vector{Float64}
    _u       :: Vector{Float64}
end

function init_pid_data(n_X::Int, n_Y::Int, n_Z::Int, their_support::Array{Tuple{Int,Int,Int}} ; tmpFloat::DataType=BigFloat) :: PID_Data
    pdummy = zeros(Float64,n_X,n_Y,n_Z)
    for xyz in their_support
        x,y,z = xyz
        @assert 1 ≤ x ≤ n_X "x out of range"
        @assert 1 ≤ y ≤ n_Y "y out of range"
        @assert 1 ≤ z ≤ n_Z "z out of range"

        pdummy[ x,y,z ] = 1.
    end

    myeval = create_My_Eval(pdummy,tmpFloat)

    resu = PID_Results( myeval.xyz, zeros(myeval.n), zeros(myeval.n), ones(myeval.n) )
    #                    support         q                ∇               p

    _lb = zeros(Float64,myeval.m)
    _ub = zeros(Float64,myeval.m)
    _l  = zeros(Float64,myeval.n)
    _u  = [Inf  for j=1:myeval.n]

    return PID_Data( 1/log(2), myeval, resu ,  pdummy, _lb, _ub, _l, _u)
    ;      #         bits                      _space
end

function pid!(pd::PID_Data, p::Vector{Float64} ) :: Void
    @assert length(p) == pd.myeval.n   "Length of p-vector doesn't match PID_Data"

    const n = pd.myeval.n
    const m = pd.myeval.m

    solver = Mosek.MosekSolver(MSK_IPAR_INTPNT_MULTI_THREAD=0, MSK_IPAR_INTPNT_MAX_ITERATIONS=500, MSK_DPAR_OPTIMIZER_MAX_TIME=100.0)

    const model = MathProgBase.NonlinearModel( solver )

    for i in 1:n
        x,y,z = pd.myeval.xyz[i]
        pd._space[ x,y,z ]  =  p[i]
    end

    set_rhs!(pd.myeval,pd._space)

    MathProgBase.loadproblem!(model, n, m, pd._l, pd._u, pd._lb, pd._ub, :Max, pd.myeval)

    MathProgBase.optimize!(model)

    if MathProgBase.status(model) ∈ [:Solve_Succeeded, :Optimal, :NearOptimal, :Suboptimal, :FeasibleApproximate]
        my_sol  = MathProgBase.getsolution(model)
        my_dual = MathProgBase.getconstrduals(model)

        for i in 1:m
            x,y,z = support[i]
            pd.results.q[i]   =   my_sol[   pd.myeval.varidx[x,y,z]   ]
        end

        pd.results.λ .= 0.
        for k in 1:ev.m
            mr = pd.myeval.mr_eq[k]
            if mr[1] == "xy"
                x,y = mr[2], mr[3]
                for z in 1:pd.myeval.n_z
                    i = varidx[x,y,z]
                    i==0  || (  pd.results.λ[i] += my_dual[k]  )
                end
            elseif mr[1]=="xz"
                x,z = mr[2], mr[3]
                for y in  1:pd.myeval.n_y
                    i = varidx[x,y,z]
                    i==0  || (  pd.results.λ[i] += my_dual[k]  )
                end
            else
                @assert false
            end
        end
    else
        # Problematic: status == :UserLimit -- and of course all other kinds of crashes
        @show status(model)
        @assert status(model) ∈ [:Solve_Succeeded, :Optimal, :NearOptimal, :Suboptimal, :FeasibleApproximate]
    end
    ;
end #^ pid!()

end #module Opt_PID
