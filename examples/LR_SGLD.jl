using Statistics

# Using Logistic Regression

function σ(x)
    1 / (1 + exp(-x))
end


function logσ(x) # approximation
    if x > -30
        return -log(1+exp(-x))
    else
        return -x
    end    
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
    onehot, N_up_to = encode(z, n_classes)
    sum(diag(onehot' * logσ.(β*X) .+ N_up_to' * (-β*X + logσ.(β*X))))
end


function predict_SGLD(βs, X)
    πs = zeros(Float64, size(βs, 1), size(X, 2), size(βs, 2)+1)
    for b = 1:size(βs,1)
        X_augmented = [X; ones(Float64, 1, size(X, 2))]
        ν = σ.(X_augmented' * βs[b,:,:]')
        comp_ν = 1 .- ν
        πs[b, :, 1] = ν[:, 1]
        for i = 2:size(ν, 2)
            πs[b, :, i] = ν[:, i] .* prod(comp_ν[:, 1:i-1], dims=2)[:, 1]
        end
        πs[b, :, size(πs, 3)] = prod(comp_ν, dims=2)[:, 1]
    end
    return Statistics.mean(πs, dims=1)[1,:,:]
end


function accuracy_SGLD(βs, X, z)
    p = predict_SGLD(βs,X)
    maxpred = mapslices(argmax, p, dims=2)[:,1]
    sum(maxpred .== z) / length(z)
end


function cross_entropy_SGLD(βs, X, z)
    p = predict_SGLD(βs,X)
    onehot = zeros(Float64, size(z,1), n_classes)
    for i = 1:n_classes
        onehot[:,i] = (z .== i)
    end
    -sum(log.(p) .* onehot) / length(z)
end



function train_SGLD(X, z, n_classes, steps, lrate, n_samples)
    X̂ = [X; ones(Float64, 1, size(X, 2))] # add 1's to account for the bias
    onehot, N_up_to = encode(z, n_classes)
    
    # Contains terms of derivative corresponding to log(σ(β'x_d)) for data point x_d
    deriv_sig_for_all = zeros(n_classes-1, size(X̂, 2), size(X̂, 1))
    # Contains terms of derivative corresponding to log(1 - σ(β'x_d)) for data point x_d
    deriv_comp_sig_for_all = zeros(n_classes-1, size(X̂, 2), size(X̂, 1))

    βs = zeros(Float64, n_samples, steps+1, n_classes-1, size(X̂, 1))

    for n = 1:n_samples
        β = βs[n,1,:,:]
        for step = 1:steps
            # Populate for all data points
            for d = 1:size(X̂, 2)
                deriv_sig_for_all[:, d, :] = deriv_sig(β, X̂[:,d])
                deriv_comp_sig_for_all[:, d, :] = deriv_comp_sig(β, X̂[:,d])
            end
            # Update parameters
            ϵ_t = (1+step)^(-0.55)
            η_t = sqrt(ϵ_t) * randn(size(β))
            derivatives_loglike = sum(onehot .* deriv_sig_for_all .+ N_up_to .* deriv_comp_sig_for_all, dims=2)[:,1,:]
            derivatives_prior = -β
            β = β + lrate * (0.5 * ϵ_t * (derivatives_loglike + derivatives_prior) + η_t)
            βs[n,step+1,:,:] = β
        end
    end
    
    return βs
end