module Opt_PID_data

export My_Eval, create_My_Eval,   condEntropy, ∇f

using MathProgBase
# http://mathprogbasejl.readthedocs.io/en/latest/nlp.html

struct My_Eval <: MathProgBase.AbstractNLPEvaluator
    n_x     :: Int64
    n_y     :: Int64
    n_z     :: Int64

    n     :: Int64         # number of variables
    m     :: Int64         # number of marginal equations

    varidx  :: Array{Int64,3}                   # 0 if variable not present; otherwise idx of var in {1,...,n}
    xyz     :: Vector{Tuple{Int64,Int64,Int64}} # xyz-triple of varidx

    eqidx   :: Dict{ Tuple{String,Int64,Int64},  Int64} # first idx is "xy" or "xz". Index of eqn in {1,...,m}
    mr_eq   :: Vector{ Tuple{String,Int64,Int64} }      # ("xy",x,y) / ("yz",y,z) of an eqn
    prb_xyz :: Array{Float64,3}
    marg_xy :: Array{Float64,2}
    marg_xz :: Array{Float64,2}

    rhs   :: Vector{Float64}                # m-vector
    Gt    :: SparseMatrixCSC{Float64,Int64} # G^T n x m; transpose of constraint matrix
    Gt_K  :: Vector{Int64}                  # findn()-info of G^T: rows (vars)
    Gt_L  :: Vector{Int64}                  # findn()-info of G^T: columns (equations)

    TmpFloat       :: DataType
    bigfloat_nbits :: Int64
end

function condEntropy{TFloat_2,TFloat}(e::My_Eval, p::Vector{TFloat_2}, dummy::TFloat)   :: TFloat
    # m_yz = marg_yz(e,p,dummy)
    s::TFloat = TFloat(0.)
    for y = 1:e.n_y
        for z = 1:e.n_z
            P_yz = TFloat(0.)
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    P_yz += p[i]
                end
            end
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    p_xyz = p[i]
                    s  +=  (  (p_xyz ≤ 0 || P_yz  ≤ 0)  ?   TFloat(0.)   :   - p_xyz*log( p_xyz / P_yz )   )
                end
            end
        end
    end
    return s
    ;
end


function ∇f{TFloat,TFloat_2}(e::My_Eval, grad::Vector{TFloat_2}, p::Vector{TFloat_2}, dummy::TFloat) :: Void
    for y = 1:e.n_y
        for z = 1:e.n_z
            # make marginal P(*yz)
            P_yz::TFloat = TFloat(0.)
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    P_yz += p[i]
                end
            end
            # make log-expressions  log( P(xyz) / P(*yz) )
            for x = 1:e.n_x
                i = e.varidx[x,y,z]
                if i>0
                    P_xyz::TFloat = TFloat( p[i] )
                    grad[i] = TFloat_2(   (P_xyz ≤ 0 || P_yz ≤ 0) ?  log(TFloat(e.n_x))  : -log( P_xyz / P_yz )  )
                end
            end
        end# for y
    end# for x
    ;
end # ∇f()



function create_My_Eval(q::Array{Float64,3}, tmpFloat::DataType, bigfloat_precision=256)
    const n_x::Int64 = size(q,1);
    const n_y::Int64 = size(q,2);
    const n_z::Int64 = size(q,3);

    # Create marginals
    prb_xyz::Array{Float64,3}  = zeros(n_x,n_y,n_z)
    marg_xy::Array{Float64,2}  = zeros(n_x,n_y)
    marg_xz::Array{Float64,2}  = zeros(n_x,n_z)

    for x in 1:n_x
        for y in 1:n_y
            for z in 1:n_z
                marg_xy[x,y] += q[x,y,z]
                marg_xz[x,z] += q[x,y,z]
            end
        end
    end
    # Find the variables
    varidx::Array{Int64,3}                   = zeros(Bool,size(q));
    xyz   = Vector{Tuple{Int64,Int64,Int64}}() # = [ (0,0,0) for i in 1:n_x*n_y*n_z ]
    sizehint!(xyz, n_x*n_y*n_z)
    for x in 1:n_x
        for y in 1:n_y
            for z in 1:n_z
                if marg_xy[x,y] > 0  &&  marg_xz[x,z] > 0
                    push!(xyz, (x,y,z))
                    varidx[x,y,z] = endof(xyz)
                    prb_xyz[x,y,z]= q[x,y,z]
                else
                    varidx[x,y,z] = 0
                end#if
            end
        end
    end
    const n::Int64 = length(xyz)


    # Find the equations
    eqidx = Dict{ Tuple{String,Int64,Int64},Int64}() # first idx is "xy" or "xz"
    mr_eq ::Vector{ Tuple{String,Int64,Int64} }   = [ ("",0,0)   for i in 1:n_x*(n_y+n_z) ]
    m::Int64 = 0
    for x in 1:n_x
        for y in 1:n_y
            if marg_xy[x,y] > 0
                m += 1
                eqidx["xy",x,y] = m
                mr_eq[m]        = ("xy",x,y)
            else
                eqidx["xy",x,y] = 0
            end#if
        end
        for z in 1:n_z
            if marg_xz[x,z] > 0
                m += 1
                eqidx["xz",x,z] = m
                mr_eq[m]        = ("xz",x,z)
            else
                eqidx["xz",x,z] = 0
            end#if
        end
    end #for x

    denseGt :: Array{Float64,2} = zeros(n,m)
    for l in 1:n
        (x,y,z) = xyz[l]
        for k in 1:m
            mr = mr_eq[k]
            if mr[1] == "xy"
                xy = mr[2:3]
                if xy[1]==x && xy[2]==y
                    denseGt[l,k] = 1.
                end
            elseif mr[1]=="xz"
                xz = mr[2:3]
                if xz[1]==x && xz[2]==z
                    denseGt[l,k] = 1.
                end
            else
                print("Fuck! Bug!")
                return;
            end
        end
    end

    Gt::SparseMatrixCSC{Float64,Int64} = sparse(denseGt)
    local Gt_K::Array{Int64,1}
    local Gt_L::Array{Int64,1}
    (Gt_K,Gt_L) = findn(Gt)

    TmpFloat       :: DataType  = tmpFloat
    bigfloat_nbits :: Int64     = bigfloat_precision

    rhs::Vector{Float64} = Vector{Float64}(m)

    return My_Eval(n_x,n_y,n_z, n,m, varidx,xyz, eqidx,mr_eq, prb_xyz, marg_xy,marg_xz, rhs, Gt,Gt_K,Gt_L,  TmpFloat,bigfloat_nbits)
    ;
end #^ create_My_eval()

function set_rhs!(ev::My_Eval, p::Array{Float64})::Void
    # Clean out old marginals
    for k in 1:ev.m
        mr = ev.mr_eq[k]
        if mr[1] == "xy"
            ev.marg_xy[ mr[2], mr[3] ] = 0.
        elseif mr[1]=="xz"
            ev.marg_xz[ mr[2], mr[3] ] = 0.
        else
            @assert false
        end
    end

    # Compute new marginals
    for (i,xyz) in enumerate(ev.xyz)
        x,y,z = xyz
        ev.marg_xy[ x,y ]   += max(0.,  p[ x,y,z ]  )
        ev.marg_xz[ x,z ]   += max(0.,  p[ x,y,z ]  )
    end

    # Fill RHS vector
    ev.rhs .= 0.
    for k in 1:ev.m
        mr = ev.mr_eq[k]
        if mr[1] == "xy"
            ev.rhs[k] = max(1.e-300,   ev.marg_xy[ mr[2], mr[3] ]   )  # the max is here because ...
        elseif mr[1]=="xz"
            ev.rhs[k] = max(1.e-300,   ev.marg_xz[ mr[2], mr[3] ]   )  # ... marginals must not be 0
        else
            @assert false
        end
    end #for all marg equations
    ;
end

end #module Opt_PID_data
