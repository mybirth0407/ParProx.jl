module ProxCox

import Base: length

export Penalty, NormL1, GroupNormL2, value, prox!
export power, CoxUpdate, CoxVariables, reset!,  get_objective!, fit!

using LinearMaps

const MapOrMatrix{T} = Union{LinearMap{T},AbstractMatrix{T}}

include("utils.jl")
include("penalties.jl")
include("cox.jl")
include("logistic.jl")
include("softmax.jl")
include("cv.jl")
using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda/utils.jl")
        include("cuda/penalties.jl")
        include("cuda/cox.jl")
    end
end

end # module
