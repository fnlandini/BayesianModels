# Logistic Regression using the Maximum Likelihood Estimate.
#

function σ(x)
    1 / (1 + exp(-x))
end

function encode(z, n_classes)
    # For the purpose of operating with matrices the function encodes the vector of classes as:
    # onehot: one-hot encoding for the classes, where the class 'n_classes' is encoded as all 0's
    # N_up_to: for each sample of class c, row i is 1 if i>=c and 0 otherwise. Note that if c=n_classes, all rows are 0's
    
    onehot = zeros(Int64, n_classes - 1, length(z))
    for i = 1:n_classes-1
        onehot[i, z .== i] .= 1
    end

    N_up_to = zeros(Int64, n_classes - 1, length(z))
    for i = 1:n_classes-1
        N_up_to[i, z .> i] .= 1
    end

    onehot, N_up_to
end


function deriv_sig(β, x)
    (1 .- σ.(β*x)) * x'
end


function deriv_comp_sig(β, x)
    .- σ.(β*x) * x'
end


function loglikelihood(n_classes, z, X, β)
    ν = σ.(β*X)
    onehot, N_up_to = encode(z, n_classes)
    sum(diag(onehot' * log.(ν) .+ N_up_to' * log.(1 .- ν)))
end


function predict_MLE(β, X)
    X_augmented = [X; ones(Float64, 1, size(X, 2))]
    ν = σ.(X_augmented' * β')
    comp_ν = 1 .- ν
    π = zeros(Float64, size(ν, 1), size(ν, 2)+1)
    π[:, 1] = ν[:, 1]
    for i = 2:size(ν, 2)
        π[:, i] = ν[:, i] .* prod(comp_ν[:, 1:i-1], dims=2)[:, 1]
    end
    π[:, size(π, 2)] = prod(comp_ν, dims=2)[:, 1]
    return π
end


function accuracy_MLE(β, X, z)
    p = predict_MLE(β, X)
    maxpred = mapslices(argmax, p, dims=2)[:,1]
    sum(maxpred .== z) / length(z)
end


function cross_entropy_MLE(β, X, z)
    p = predict_MLE(β, X)
    onehot = zeros(Float64, size(z,1), n_classes)
    for i = 1:n_classes
        onehot[:,i] = (z .== i)
    end
    -sum(log.(p) .* onehot) / length(z)
end


function train_MLE(X, z, n_classes, steps, η, optimizer)
    X̂ = [X; ones(Float64, 1, size(X, 2))] # add 1's to account for the bias
    onehot, N_up_to = encode(z, n_classes)

    βs = zeros(Float64, steps+1, n_classes-1, size(X̂, 1))
    β = βs[1, :, :]
    # Contains terms of derivative corresponding to log(σ(β'x_d)) for data point x_d
    deriv_sig_for_all = zeros(size(β, 1), size(X̂, 2), size(X̂, 1))
    # Contains terms of derivative corresponding to log(1 - σ(β'x_d)) for data point x_d
    deriv_comp_sig_for_all = zeros(size(β, 1), size(X̂, 2), size(X̂, 1))

    for step = 1:steps
        # Populate for all data points
        for d = 1:size(X̂, 2)
            deriv_sig_for_all[:, d, :] = deriv_sig(β, X̂[:,d])
            deriv_comp_sig_for_all[:, d, :] = deriv_comp_sig(β, X̂[:,d])
        end
        # Update parameters
        derivatives = sum(onehot .* deriv_sig_for_all .+ N_up_to .* deriv_comp_sig_for_all, dims=2)[:,1,:]
        if optimizer == "SGD"
            β = β + η * derivatives
        elseif optimizer == "Adam"
            if !(@isdefined m)
                m = zeros(Float64, size(derivatives, 1), size(derivatives, 2))
                v = zeros(Float64, size(derivatives, 1), size(derivatives, 2))
            end
            ϵ = 1e-8
            β_1 = 0.9
            β_2 = 0.999
            m = β_1 * m .+ (1-β_1) * derivatives
            v = β_2 * v .+ (1-β_2) * derivatives.^2
            m̂ = m ./ (1 - β_1^step)
            v̂ = v ./ (1 - β_2^step)
            β = β + η * m̂ ./ (sqrt.(v̂) .+ ϵ)
        end
        βs[step+1, :, :] = β
    end
    return βs
end

