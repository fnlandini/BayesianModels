# Logistic Regression using Variational Bayes with reparameterization trick with diagonal covariance matrix

function deriv_sig_mean(μ, λ, ϵ, x)
    (1 .- σ.((μ + exp.(.5 * λ).*ϵ) * x)) .* x'
end


function deriv_comp_sig_mean(μ, λ, ϵ, x)
    .- σ.((μ + exp.(.5 * λ).*ϵ) * x) .* x'
end


function deriv_sig_cov(μ, λ, ϵ, x)
    (1 .- σ.((μ + exp.(.5 * λ).*ϵ) * x)) .* (ϵ' .* x)' * 0.5
end


function deriv_comp_sig_cov(μ, λ, ϵ, x)
    .- σ.((μ + exp.(.5 * λ).*ϵ) * x) .* (ϵ' .* x)' * 0.5
end


function loglikelihood(n_classes, z, X, μ, λ)
    ν = zeros(Float64, n_classes-1, size(X, 2))
    for i = 1:(n_classes-1)
        ϵi = randn(size(μ, 2))
        βi = μ[i,:] + exp.(.5 * λ[i,:]) .* ϵi
        ν[i, :] = σ.(βi'*X)
    end
    onehot, N_up_to = encode(z, n_classes)
    logParams = log.(ν)
    logComplementParams = log.(1 .- ν)
    sum(diag(onehot' * logParams .+ N_up_to' * logComplementParams))
end


function KL_divergence(μ, λ)
    sum(0.5 .* sum(exp.(λ) .+ μ.^2 .- λ .- 1))
end


# Predict the parameters using Maximum A Posteriori (MAP)
function predict_map(μ, X::Matrix{T}) where T <: AbstractFloat
    X_augmented = [X; ones(Float64, 1, size(X, 2))]
    ν = σ.(X_augmented' * μ')
    comp_ν = 1 .- ν
    π = zeros(Float64, size(ν, 1), size(ν, 2)+1)
    π[:, 1] = ν[:, 1]
    for i = 2:size(ν, 2)
        π[:, i] = ν[:, i] .* prod(comp_ν[:, 1:i-1], dims=2)[:, 1]
    end
    π[:, size(π, 2)] = prod(comp_ν, dims=2)[:, 1]
    return π
end


function accuracy_map(μ, X, z)
    p = predict_map(μ, X)
    maxpred = mapslices(argmax, p, dims=2)[:,1]
    sum(maxpred .== z) / length(z)
end


# Predict the parameters using the marginal posterior
function predict_marginal(μ, λ, X)
    # Following https://arxiv.org/pdf/1703.00091.pdf, we use the
    # approximation:
    # ⟨σ(ψ)⟩ ≈ σ(μ / √(1 + a * σ²))
    # where a = 0.368
    a = 0.368
    # In Bishop 4.5.2 a very similar approximation uses a = π / 8 ≈ 0.392...
    X_augmented = [X; ones(Float64, 1, size(X, 2))]
    Σ = zeros(Float64, size(λ, 1), size(λ, 2), size(λ, 2))
    ν = zeros(Float64, size(X, 2), size(μ, 1))
    for i = 1:size(λ, 1)
        Σ[i,:,:] = diagm(exp.(λ[i,:]))
        E_ββᵀ = Σ[i,:,:] + μ[i,:] * μ[i,:]'
        ψ_μ = X_augmented' * μ[i,:]
        ψ² = dropdims(sum(X_augmented .* (E_ββᵀ * X_augmented), dims = 1), dims = 1)
        ψ_σ² = ψ² .- (ψ_μ .^ 2)
        y = ψ_μ ./ sqrt.(1 .+ a * ψ_σ²)
        ν[:,i] = 1 ./ (1 .+ exp.(-y))
    end
    
    comp_ν = 1 .- ν
    π = zeros(Float64, size(ν, 1), size(ν, 2)+1)
    π[:, 1] = ν[:, 1]
    for i = 2:size(ν, 2)
        π[:, i] = ν[:, i] .* prod(comp_ν[:, 1:i-1], dims=2)[:, 1]
    end
    π[:, size(π, 2)] = prod(comp_ν, dims=2)[:, 1]
    return π
end


function accuracy_marginal(μ, λ, X, z)
    p = predict_marginal(μ, λ, X)
    maxpred = mapslices(argmax, p, dims=2)[:,1]
    sum(maxpred .== z) / length(z)
end


function cross_entropy_marginal(μ, λ, X, z)
    p = predict_marginal(μ, λ, X)
    onehot = zeros(Float64, size(z,1), n_classes)
    for i = 1:n_classes
        onehot[:,i] = (z .== i)
    end
    -sum(log.(p) .* onehot) / length(z)
end


function train_VB_rep_trick_diag(X, z, n_classes, steps, η, optimizer, rep_trick_n_samples)
    X̂ = [X; ones(Float64, 1, size(X, 2))] # add 1's to account for the bias
    onehot, N_up_to = encode(z, n_classes)
    
    # Parameters of the model
    μs = zeros(Float64, steps+1, n_classes-1, size(X̂, 1))
    λs = ones(Float64, steps+1, n_classes-1, size(X̂, 1))
                 # since the covariance matrix is diagonal, 
                 # only the elements in the diagonal are modeled. 
                 # Also, in order to have a positive definite matrix, 
                 # the elements of the diagonal of Σ are exp.{λ}
    μ = μs[1, :, :]
    λ = λs[1, :, :]

    L = rep_trick_n_samples # number of samples to use

    # Contains terms of derivative of mean corresponding to log(σ(β'x_d)) for data point x_d and sample l
    deriv_sig_for_all_mean = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)
    # Contains terms of derivative of covariance corresponding to log(σ(β'x_d)) for data point x_d and sample l
    deriv_sig_for_all_cov = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)

    # Contains terms of derivative of mean corresponding to log(1 - σ(β'x_d)) for data point x_d and sample l
    deriv_comp_sig_for_all_mean = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)
    # Contains terms of derivative of mean corresponding to log(1 - σ(β'x_d)) for data point x_d and sample l
    deriv_comp_sig_for_all_cov = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)

    for step = 1:steps    
        # Populate for all data points and samples
        for l = 1:L
            ϵl = randn(n_classes-1, size(X_train, 1)+1)
            for d = 1:size(X̂, 2)
                deriv_sig_for_all_mean[:, d, :, l] = deriv_sig_mean(μ, λ, ϵl, X̂[:,d])
                deriv_comp_sig_for_all_mean[:, d, :, l] = deriv_comp_sig_mean(μ, λ, ϵl, X̂[:,d])
                deriv_sig_for_all_cov[:, d, :, l] = deriv_sig_cov(μ, λ, ϵl, X̂[:,d])
                deriv_comp_sig_for_all_cov[:, d, :, l] = deriv_comp_sig_cov(μ, λ, ϵl, X̂[:,d])
            end
        end

        # Update parameters
        derivatives_mean = 1.0/L .* sum(onehot .* deriv_sig_for_all_mean .+ N_up_to .* deriv_comp_sig_for_all_mean, dims=[2,4])[:,1,:,1]
        # Subtract KL divergence term
        derivatives_mean .-= μ
        if optimizer == "SGD"
            μ = μ + η * derivatives_mean
        elseif optimizer == "Adam"
            if !(@isdefined m_μ)
                m_μ = zeros(Float64, size(derivatives_mean, 1), size(derivatives_mean, 2))
                v_μ = zeros(Float64, size(derivatives_mean, 1), size(derivatives_mean, 2))
            end
            ϵ_μ = 1e-8
            β_1 = 0.9
            β_2 = 0.999
            m_μ = β_1 * m_μ .+ (1-β_1) * derivatives_mean
            v_μ = β_2 * v_μ .+ (1-β_2) * derivatives_mean.^2
            m̂_μ = m_μ ./ (1 - β_1^step)
            v̂_μ = v_μ ./ (1 - β_2^step)
            μ = μ + η * m̂_μ ./ (sqrt.(v̂_μ) .+ ϵ_μ)
        end

        derivatives_cov = 1.0/L .* sum(onehot .* deriv_sig_for_all_cov .+ N_up_to .* deriv_comp_sig_for_all_cov, dims=[2,4])[:,1,:,1]
        # Subtract KL divergence term
        derivatives_cov .-= 0.5 .* (exp.(λ) .- 1)
        if optimizer == "SGD"
            λ = λ + η * derivatives_cov
        elseif optimizer == "Adam"
            if !(@isdefined m_λ)
                m_λ = zeros(Float64, size(derivatives_cov, 1), size(derivatives_cov, 2))
                v_λ = zeros(Float64, size(derivatives_cov, 1), size(derivatives_cov, 2))
            end
            ϵ_λ = 1e-8
            β_1 = 0.9
            β_2 = 0.999
            m_λ = β_1 * m_λ .+ (1-β_1) * derivatives_cov
            v_λ = β_2 * v_λ .+ (1-β_2) * derivatives_cov.^2
            m̂_λ = m_λ ./ (1 - β_1^step)
            v̂_λ = v_λ ./ (1 - β_2^step)
            λ = λ + η * m̂_λ ./ (sqrt.(v̂_λ) .+ ϵ_λ)
        end
        
        μs[step+1, :, :] = μ
        λs[step+1, :, :] = λ
    end

    return μs, λs
end