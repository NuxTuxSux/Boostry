using StatsBase, LinearAlgebra
import StatsBase.fit!
###### CONSTANTS
# scaling factor in similarity function
α = 1
α₀ = .1
α₁ = 3.



"""
`δ(v::Vector, w::Vector)`

Gives square distance between `v` and `w`
"""
@inline δ(v::Vector{Float64},w::Vector{Float64}) = sum((v - w) .^ 2)


"""
`neighbours(data::Matrix, i::Int, k::Int)`

Gives nearest `k` rows of `data` to its `i` row as an array of `(rowindex,squarecorrecteddistance)` couples.
Squared corrected distance is squared distance divided by the dimension `n`
"""
function neighbours(data::Matrix, v::Vector, k::Int = 20)
    sqdist = [(j,δ(data[j,:],v)) for j in 1:(size(data)[1])]
    res = sort(sqdist, by = x -> x[2])[1:k]
    first.(res), [x[2]/length(v) for x ∈ res]
end



mutable struct RBoost
    models::Array
    #names::Vector{String}
    pretest::Union{Nothing,Matrix}
    pretesttarget::Union{Nothing,Vector}
    # ε[i,j] gives error of model i on pretest point j
    ε::Union{Nothing,Matrix{Float64}}
    # ho messo l'esponente per la distance2similarity parametrico
    α::Union{Nothing,Float64}
    RBoost(models::Array) = new(models, nothing, nothing, nothing, nothing)
end

mutable struct CBoost
    regressor::RBoost
    activation::Function
    threshold::Union{Nothing,Float64}
    CBoost(models::Array, activation::Union{Nothing,Function} = tanh) = new(RBoost(models), activation, nothing)
end

function fit!(B::RBoost, data::Matrix, target::Vector; α₀ = α₀, α₁ = α₁)
    ## eventualmente provare il LOO per non necessitare di pretest e usare tutto il test per le predizioni. sistemare dettagli

    # costants
    pretestsplit = .3
    
    # split the train dataset to get pretest sample
    pretest = sample(1:size(data)[1],Int(floor(size(data)[1]*pretestsplit)),replace = false)
    ε = zeros(length(B.models),length(pretest))
    
    train = setdiff(1:size(data)[1], pretest)
    # train = 1:size(data)[1]
    for (ixmodel,model) in enumerate(B.models)
        println("Fitting $(string(model))")
        model.fit(data[train,:], reshape(target[train,:],(:,)))
        ε[ixmodel,:]  = abs.(model.predict(data[pretest,:]) - target[pretest])
    end

    B.pretest = data[pretest,:]
    B.pretesttarget = target[pretest]
    B.ε = ε;

    println("α tuning")
    setalpha!(B)
    # errs = []
    # for α ∈ α₀:.01:α₁
        # err = sum(abs.(predict(B,data[pretest,:],α) .- target[pretest]))
        # push!(errs,err)
        # println("α = $α: $err")
    # end
    # findmin(errs)
end

function setalpha!(B::RBoost, α₀ = 0.01, α₁ = 5., n_iter = 14)
    for _ in 1:n_iter
        α = (α₀ + α₁) / 2
        err₀ = sum(abs.(predict(B,B.pretest,α₀) .- B.pretesttarget))
        err₁ = sum(abs.(predict(B,B.pretest,α₁) .- B.pretesttarget))
        err = sum(abs.(predict(B,B.pretest,α) .- B.pretesttarget))
        α₀, α₁ = [x[2] for x in sort([(err₀,α₀),(err₁,α₁),(err,α)], by = first)[1:2]]
    end
    B.α = (α₀ + α₁)/2
end

function fit!(B::CBoost, data::Matrix, target::Vector)
    # codifico false:-1, true:1
    fit!(B.regressor,data,target)
    calcThreshold!(B, data, target)
end


## controllare
@inline distance2similarity(d, α) = (d+.01)^(-α)

@inline activation(x) = tanh(x)

function predict(B::RBoost, datarow::Vector, α = α, verbose::Bool = false)
    # take the double weighted mean by inversedinstance and singlemodelerror of predicted values
    ## volendo se li prendo tutti posso scriverlo in modo matriciale
    nbIxs, nbDist = neighbours(B.pretest,datarow)
    # calcolo le predizioni dei modelli
    predictions = [first(model.predict(reshape(datarow,(1,:)))) for model in B.models]

    # se ho già calcolato α la utilizzo
    if !isnothing(B.α)
        α = B.α
    end

    # calcolo i pesi da dare ad ogni modello in base al punto attuale
    Λ = [sum(distance2similarity.(nbDist .* B.ε[mdl,nbIxs], α)) for mdl ∈ 1:length(B.models)]
    Λ /= sum(Λ)

    # calcolo la predizione
    if verbose
        println("prova")
        (
            prediction = sum(Λ.*predictions),
            weights = Λ,
            original_prediction = predictions
        )
    else
        sum(Λ.*predictions)
    end
end

predict(B::RBoost, data::Matrix, α = α, verbose::Bool = false) = [predict(B,data[r,:], α, verbose) for r ∈ 1:(size(data)[1])]

predict(B::CBoost, datarow::Vector, α = α) = 2sign(B.activation(predict(B.regressor, datarow, α)) > B.threshold) .- 1

predict(B::CBoost, data::Matrix, α = α) = [predict(B,data[r,:], α) for r ∈ 1:(size(data)[1])]
# chiamare nel fit dei CBoost
function calcThreshold!(B::CBoost, train::Matrix, target::Vector, α = α)
    predvalues = predict(B.regressor,train, α)
    guesses = 0:.01:1
    guesses = sample(guesses, length(guesses),replace=false)
    B.threshold = guesses[findmax([count(sign.(B.activation.(predvalues) .- thrsh) .== target) for thrsh ∈ guesses])[2]]
end







#= TOTÒ
    similarità
· provare similarità con gaussiana

· LOO   (forse computazionalmente da evitare, altrimenti trovo prima il parametro e faccio uno StochasticLOO!)
· altrimenti usare un il training anche per la parte di pretesting
· sistemare gli iperparametri ché così fanno un po' schifo
· scrivere una fase due in cui alleno i modelli su dataset su cui hanno sbagliato meno
· sistemare distance2similarity, e anche un po' tutto il resto
· provare a rendere parametrica la distanza, per fare tuning su quel parametro
· fare il tuning di α per bisezione
=#




### Test per attribuire metodi alle struct, eventualmente lo farò più in là
mutable struct C
    x::Int
    stampa::Function
    function C(a::Int)
        r = new(a,()->nothing)
        r.stampa = () -> r.x
        r
    end
end

mutable struct D
    x::Int
    function stampa() x; end
    function aggiorna() x = v; end
end

