# Logistic Regression using Variational Bayes with reparameterization trick with diagonal covariance matrix

function deriv_sig_mean(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(μ,1), size(μ,2))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:] = (1 - σ((μ[i,:] + A*ϵ[i,:])' * x)) .* x'
    end
    d
end


function deriv_comp_sig_mean(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(μ,1), size(μ,2))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:] = - σ((μ[i,:] + A*ϵ[i,:])' * x) .* x'
    end
    d
end


function deriv_sig_U(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(U,1), size(U,2), size(U,3))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:,:] = (1 - σ((μ[i,:] + A*ϵ[i,:])' * x)) .* (ϵ[i,:]*x' + x*ϵ[i,:]')'
    end
    d
end


function deriv_comp_sig_U(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(U,1), size(U,2), size(U,3))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:,:] = - σ((μ[i,:] + A*ϵ[i,:])' * x) .* (ϵ[i,:]*x' + x*ϵ[i,:]')'
    end
    d
end


function deriv_sig_λ(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(λ,1), size(λ,2))
    for i = 1:size(λ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:] = (1 - σ((μ[i,:] + A*ϵ[i,:])' * x)) .* (ϵ[i,:] .* x)
    end
    d
end


function deriv_comp_sig_λ(μ, U, λ, ϵ, x)
    d = zeros(Float64, size(λ,1), size(λ,2))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(λ[i,:]))
        d[i,:] = - σ((μ[i,:] + A*ϵ[i,:])' * x) .* (ϵ[i,:] .* x)
    end
    d
end


function loglikelihood(n_classes, z, X, μ, U, λ)
    ν = zeros(Float64, n_classes-1, size(X, 2))
    for i = 1:(n_classes-1)
        ϵi = randn(size(μ, 2))
        βi = μ[i,:] + (U[i,:,:] + diagm(exp.(λ[i,:]))) * ϵi
        ν[i, :] = σ.(βi'*X)
    end
    onehot, N_up_to = encode(z, n_classes)
    logParams = log.(ν)
    logComplementParams = log.(1 .- ν)
    sum(diag(onehot' * logParams .+ N_up_to' * logComplementParams))
end


function KL_divergence(μ, U, λ)
    d = zeros(Float64, size(μ,1))
    for i = 1:size(μ,1)
        A = U[i,:,:] + diagm(exp.(.5 * λ[i,:]))
        d[i] = 0.5 * (tr(A'*A) + μ[i,:]'*μ[i,:] - size(μ,2) - log(det(A'*A)))
    end
    sum(d)
end


# Predict the classes using the Maximum A Posteriori (MAP) parameters
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


function predict_marginal(μ, U, λ, X)
    # Following https://arxiv.org/pdf/1703.00091.pdf, we use the
    # approximation:
    # ⟨σ(ψ)⟩ ≈ σ(μ / √(1 + a * σ²))
    # where a = 0.368
    a = 0.368
    # In Bishop 4.5.2 a very similar approximation uses a = π / 8 ≈ 0.392...
    X_augmented = [X; ones(Float64, 1, size(X, 2))]
    Σ = zeros(Float64, size(U, 1), size(U, 2), size(U, 3))
    ν = zeros(Float64, size(X, 2), size(μ, 1))
    for i = 1:size(U, 1)
        A = U[i,:,:] + diagm(exp.(.5 * λ[i,:]))
        Σ[i,:,:] = A'*A
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


function accuracy_marginal(μ, U, λ, X, z)
    p = predict_marginal(μ, U, λ, X)
    maxpred = mapslices(argmax, p, dims=2)[:,1]
    sum(maxpred .== z) / length(z)
end


function cross_entropy_marginal(μ, U, λ, X, z)
    p = predict_marginal(μ, U, λ, X)
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
    Us = zeros(Float64, steps+1, n_classes-1, size(X̂, 1), size(X̂, 1))
    λs = ones(Float64, steps+1, n_classes-1, size(X̂, 1))
    # We define D = diagm(exp(λ)), A = U + D and Σ = AtA
    # where U is upper triangular with 0's in the diagonal and the diagonal matrix D has all positive values in the diagonal
    μ = μs[1, :, :]
    U = Us[1, :, :, :]
    λ = λs[1, :, :]
    
    L = rep_trick_n_samples # number of samples to use

    # Contains terms of derivative of mean corresponding to log(σ(β'x_d)) for data point x_d and sample l
    deriv_sig_for_all_mean = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)
    # Contains terms of derivative of U corresponding to log(σ(β'x_d)) for data point x_d and sample l
    deriv_sig_for_all_U = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), size(X̂, 1), L)
    # Contains terms of derivative of λ corresponding to log(σ(β'x_d)) for data point x_d and sample l
    deriv_sig_for_all_λ = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)

    # Contains terms of derivative of mean corresponding to log(1 - σ(β'x_d)) for data point x_d and sample l
    deriv_comp_sig_for_all_mean = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)
    # Contains terms of derivative of U corresponding to log(1 - σ(β'x_d)) for data point x_d and sample l
    deriv_comp_sig_for_all_U = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), size(X̂, 1), L)
    # Contains terms of derivative of λ corresponding to log(1 - σ(β'x_d)) for data point x_d and sample l
    deriv_comp_sig_for_all_λ = zeros(size(μ, 1), size(X̂, 2), size(X̂, 1), L)

    for step = 1:steps
        # Populate for all data points and samples
        for l = 1:L
            ϵl = randn(n_classes-1, size(X̂, 1))
            for d = 1:size(X̂, 2)
                deriv_sig_for_all_mean[:, d, :, l] = deriv_sig_mean(μ, U, λ, ϵl, X̂[:,d])
                deriv_comp_sig_for_all_mean[:, d, :, l] = deriv_comp_sig_mean(μ, U, λ, ϵl, X̂[:,d])
                deriv_sig_for_all_U[:, d, :, :, l] = deriv_sig_U(μ, U, λ, ϵl, X̂[:,d])
                deriv_comp_sig_for_all_U[:, d, :, :, l] = deriv_comp_sig_U(μ, U, λ, ϵl, X̂[:,d])
                deriv_sig_for_all_λ[:, d, :, l] = deriv_sig_λ(μ, U, λ, ϵl, X̂[:,d])
                deriv_comp_sig_for_all_λ[:, d, :, l] = deriv_comp_sig_λ(μ, U, λ, ϵl, X̂[:,d])
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

        # Compute A before updating both U and λ
        A = zeros(Float64, size(U, 1), size(U, 2), size(U, 3))
        for i = 1:size(A,1)
            A[i,:,:] = U[i,:,:] + diagm(exp.(λ[i,:]))
        end

        derivatives_U = 1.0/L .* sum(onehot .* deriv_sig_for_all_U .+ N_up_to .* deriv_comp_sig_for_all_U, dims=[2,5])[:,1,:,:,1]
        # Subtract KL divergence term
        for i = 1:size(U,1)
            derivatives_U[i,:,:] -= (A[i,:,:]' - A[i,:,:] * inv(A[i,:,:]'*A[i,:,:]))'
            # Consider only derivatives above the diagonal for the update
            derivatives_U[i,:,:] = triu(derivatives_U[i,:,:], 1)
        end
        if optimizer == "SGD"
            U = U + η * derivatives_U
        elseif optimizer == "Adam"
            if !(@isdefined m_U)
                m_U = zeros(Float64, size(derivatives_U, 1), size(derivatives_U, 2), size(derivatives_U, 3))
                v_U = zeros(Float64, size(derivatives_U, 1), size(derivatives_U, 2), size(derivatives_U, 3))
            end
            ϵ_U = 1e-8
            β_1 = 0.9
            β_2 = 0.999
            m_U = β_1 * m_U .+ (1-β_1) * derivatives_U
            v_U = β_2 * v_U .+ (1-β_2) * derivatives_U.^2
            m̂_U = m_U ./ (1 - β_1^step)
            v̂_U = v_U ./ (1 - β_2^step)
            U = U + η * m̂_U ./ (sqrt.(v̂_U) .+ ϵ_U)
        end

        derivatives_λ = 1.0/L .* sum(onehot .* deriv_sig_for_all_λ .+ N_up_to .* deriv_comp_sig_for_all_λ, dims=[2,4])[:,1,:,1]
        # Subtract KL divergence term
        for i = 1:size(λ,1)
            derivatives_λ[i,:] -= (A[i,:,:]' - A[i,:,:] * inv(A[i,:,:]'*A[i,:,:]))' * exp.(λ[i,:])
        end
        if optimizer == "SGD"
            λ = λ + η * derivatives_λ
        elseif optimizer == "Adam"
            if !(@isdefined m_λ)
                m_λ = zeros(Float64, size(derivatives_λ, 1), size(derivatives_λ, 2), size(derivatives_λ, 3))
                v_λ = zeros(Float64, size(derivatives_λ, 1), size(derivatives_λ, 2), size(derivatives_λ, 3))
            end
            ϵ_λ = 1e-8
            β_1 = 0.9
            β_2 = 0.999
            m_λ = β_1 * m_λ .+ (1-β_1) * derivatives_λ
            v_λ = β_2 * v_λ .+ (1-β_2) * derivatives_λ.^2
            m̂_λ = m_λ ./ (1 - β_1^step)
            v̂_λ = v_λ ./ (1 - β_2^step)
            λ = λ + η * m̂_λ ./ (sqrt.(v̂_λ) .+ ϵ_λ)
        end

        μs[step+1, :, :] = μ
        Us[step+1, :, :, :] = U
        λs[step+1, :, :] = λ
    end
    
    return μs, Us, λs
end