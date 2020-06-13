using StatsBase, LinearAlgebra

###### COSTANTS
# scaling factor in similarity function
σ = .11    # for absexp
σ = .0155   # for gaussian



"""
`δ(v::Vector, w::Vector)`

Gives square distance between `v` and `w`
"""
@inline δ(v::Vector{Float64},w::Vector{Float64}) = sum((v - w) .* (v - w))


"""
`neighbours(data::Matrix, i::Int, k::Int)`

Gives nearest `k` rows of `data` to its `i` row as an array of `(rowindex,squarecorrecteddistance)` couples.
Squared corrected distance is squared distance divided by the dimension `n`
"""
function neighbours(data::Matrix, v::Vector, k::Int = 101)
    sqdist = [(j,δ(data[j,:],v)) for j in 1:(size(data)[1])]
    res = sort(sqdist, by = x -> x[2])[1:k]
    first.(res), [x[2]/length(v) for x ∈ res]
end



mutable struct Boost
    models::Array
    #names::Vector{String}
    pretest::Matrix
    # ε[i,j] gives error of model i on pretest point j
    ε::Matrix{Float64}
    Boost(models::Array) = new(models,zeros(1,1),zeros(1,1))
end

function fit!(B::Boost, data::Matrix, target::Vector)
    ## eventualmente provare il LOO per non necessitare di pretest e usare tutto il test per le predizioni. sistemare dettagli

    # costants
    pretestsplit = .4
    
    # split the train dataset to get pretest sample
    pretest = sample(1:size(data)[1],Int(floor(size(data)[1]*pretestsplit)),replace = false)
    ε = zeros(length(B.models),length(pretest))
    
    train = setdiff(1:size(data)[1], pretest)
    for (ixmodel,model) in enumerate(B.models)
        println("Fitting $(string(model))")
        model.fit(data[train,:], target[train,:])
        ε[ixmodel,:]  = abs.(model.predict(data[pretest,:]) - target[pretest,:])
    end

    B.pretest = data[pretest,:]
    B.ε = ε
end

@inline distance2similarity(d, σ = σ) = 1/d

function predict(B::Boost, datarow::Vector, σ = σ, verbose::Bool = false)
    # take the double weighted mean by inversedinstance and singlemodelerror of predicted values
    ## volendo se li prendo tutti posso scriverlo in modo matriciale
    nbIxs, nbDist = neighbours(B.pretest,datarow)
    # calcolo le predizioni dei modelli
    predictions = [first(model.predict(reshape(datarow,(1,:)))) for model in B.models]

    # calcolo i pesi da dare ad ogni modello in base al punto attuale
    Λ = [sum(distance2similarity.(nbDist .* B.ε[mdl,nbIxs], σ, A)) for mdl ∈ 1:length(B.models)]
    Λ /= sum(Λ)

    # calcolo la predizione
    if verbose
        (
            prediction = sum(Λ.*predictions),
            weights = Λ,
            original_prediction = predictions
        )
    else
        sum(Λ.*predictions)
    end
end

predict(B::Boost, data::Matrix, σ = σ, verbose::Bool = false) = [predict(B,data[r,:], σ, verbose) for r ∈ 1:(size(data)[1])]








#= TOTÒ
    similarità
· provare similarità con gaussiana
· ? aggiungere scaling factor nella similarità

· LOO
· sistemare gli iperparametri ché così fanno un po' schifo
· creare classificatore
· scrivere una fase due in cui alleno i modelli su dataset su cui hanno sbagliato meno
=#




### Test per attribuire metodi alle struct
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

