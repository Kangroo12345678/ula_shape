# ==============================================================================
# Author: Michael Kang
# Empirical Investigation of ULA Bias Dependence on Convexity Parameter
# ==============================================================================
using LinearAlgebra, Statistics, Random, Distributions
using QuadGK, StatsBase, Interpolations, CSV, DataFrames, Plots
using ProgressMeter, Printf, Logging, JLD2
Random.seed!(42)
# Create directories for results if they do not exist. This ensures organized storage of outputs.
if !isdir("ula_bias_results")
    mkdir("ula_bias_results")
end
if !isdir("ula_bias_results/detailed_analysis")
    mkdir("ula_bias_results/detailed_analysis")
end
if !isdir("ula_bias_results/visualizations")
    mkdir("ula_bias_results/visualizations")
end
# ==============================================================================
# LINEAR REGRESSION FUNCTION
# ==============================================================================
# A simple struct to hold linear regression results, including slope, intercept, and R-squared.
struct LinRegressResult
    slope::Float64
    intercept::Float64
    r_squared::Float64
end
# Performs linear regression on vectors x and y. Includes a check for low variance in x to avoid numerical issues.
function linregress(x::AbstractVector, y::AbstractVector)
    if length(x) < 2 || var(x) < 1e-12
        @warn "Variance of x is near zero in linregress. Returning slope=0."
        return LinRegressResult(0.0, mean(y), 0.0)
    end
    slope = cov(x, y) / var(x)
    intercept = mean(y) - slope * mean(x)
    y_pred = slope .* x .+ intercept
    r_squared = cor(x, y)^2
    return LinRegressResult(slope, intercept, r_squared)
end
# ==============================================================================
# MODULE: ConvexityAnalysis
# ==============================================================================
module ConvexityAnalysis
using Reexport
@reexport using LinearAlgebra, Statistics, Random, Distributions, QuadGK, StatsBase,
                Interpolations, CSV, DataFrames, Plots, ProgressMeter, Printf, Logging, JLD2
export PotentialFamily, ConvexityParams, ULAParams, AnalysisResult
export LocalWeakPotential, GlobalWeakPotential, AsymmetricPotential
export compute_convexity_constants
export run_full_analysis
export analyze_alpha_dependence, analyze_convergence_rate
export visualize_potential_and_samples
# ==============================================================================
# LINEAR REGRESSION FUNCTION (Repeated in module for self-containment)
# ==============================================================================
struct LinRegressResult
    slope::Float64
    intercept::Float64
    r_squared::Float64
end
function linregress(x::AbstractVector, y::AbstractVector)
    if length(x) < 2 || var(x) < 1e-12
        @warn "Variance of x is near zero in linregress. Returning slope=0."
        return LinRegressResult(0.0, mean(y), 0.0)
    end
    slope = cov(x, y) / var(x)
    intercept = mean(y) - slope * mean(x)
    y_pred = slope .* x .+ intercept
    r_squared = cor(x, y)^2
    return LinRegressResult(slope, intercept, r_squared)
end
# ==============================================================================
# STABLE LOGCOSH FUNCTION TO AVOID OVERFLOW IN POTENTIAL COMPUTATIONS
# ==============================================================================
# Stable computation of log(cosh(x)) to prevent overflow for large |x|.
# For |x| >= 20, uses asymptotic approximation log(cosh(x)) ≈ |x| - log(2) + log1p(exp(-2|x|)).
function logcosh_stable(x::Float64)
    ax = abs(x)
    if ax < 20.0
        return log(cosh(ax))
    else
        return ax - log(2.0) + log1p(exp(-2.0 * ax))
    end
end
# ==============================================================================
# Families of Potentials with Varying Convexity Properties
# ==============================================================================
abstract type AbstractPotential end
"""
    ConvexityParams
- `alpha::Float64`: The infimum of the second derivative (convexity parameter).
- `beta::Float64`: The supremum of the second derivative (smoothness parameter).
- `R::Float64`: An effective radius for the region of weak convexity (computed approximately for new families).
"""
struct ConvexityParams
    alpha::Float64
    beta::Float64
    R::Float64
end
"""
    LocalWeakPotential(alpha, beta)
Defines a potential with weak convexity near the origin (local) and strong convexity in the tails.
V(x) = (beta/2) x^2 - (beta - alpha) log(cosh(|x|))
This ensures V''(x) = beta - (beta - alpha) sech^2(|x|), with inf V'' = alpha at x=0, sup V'' = beta at infinity.
Uses stable logcosh to avoid numerical overflow.
"""
struct LocalWeakPotential <: AbstractPotential
    alpha::Float64
    beta::Float64
end
function (p::LocalWeakPotential)(x::Float64)
    abs_x = abs(x)
    return (p.beta / 2) * abs_x^2 - (p.beta - p.alpha) * logcosh_stable(x)
end
function grad(p::LocalWeakPotential, x::Float64)
    abs_x = abs(x)
    return sign(x) * (p.beta * abs_x - (p.beta - p.alpha) * tanh(abs_x))
end
"""
    GlobalWeakPotential(alpha, beta)
Defines a potential with strong convexity near the origin and weak convexity in the tails (global).
V(x) = (alpha/2) x^2 + (beta - alpha) log(cosh(|x|))
This ensures V''(x) = alpha + (beta - alpha) sech^2(|x|), with inf V'' = alpha at infinity, sup V'' = beta at x=0.
Uses stable logcosh for consistency.
"""
struct GlobalWeakPotential <: AbstractPotential
    alpha::Float64
    beta::Float64
end
function (p::GlobalWeakPotential)(x::Float64)
    abs_x = abs(x)
    return (p.alpha / 2) * abs_x^2 + (p.beta - p.alpha) * logcosh_stable(x)
end
function grad(p::GlobalWeakPotential, x::Float64)
    abs_x = abs(x)
    return sign(x) * (p.alpha * abs_x + (p.beta - p.alpha) * tanh(abs_x))
end
"""
    AsymmetricPotential(R, alpha_left, alpha_right, beta)
As defined in the reference code, for locally non-convex cases with convexity at infinity.
Included to test the reflection coupling implications for non-convex potentials.
"""
struct AsymmetricPotential <: AbstractPotential
    R::Float64
    alpha_left::Float64
    alpha_right::Float64
    beta::Float64
end
function (p::AsymmetricPotential)(x::Float64)
    if x < -p.R
        c_left = -p.beta * cos(π * (-p.R))
        d_left = p.beta * π * sin(π * (-p.R))
        return 0.5 * p.alpha_left * (x + p.R)^2 + d_left * (x + p.R) + c_left
    elseif x > p.R
        c_right = -p.beta * cos(π * p.R)
        d_right = p.beta * π * sin(π * p.R)
        return 0.5 * p.alpha_right * (x - p.R)^2 + d_right * (x - p.R) + c_right
    else
        return -p.beta * cos(π * x)
    end
end
function grad(p::AsymmetricPotential, x::Float64)
    if x < -p.R
        d_left = p.beta * π * sin(π * (-p.R))
        return p.alpha_left * (x + p.R) + d_left
    elseif x > p.R
        d_right = p.beta * π * sin(π * p.R)
        return p.alpha_right * (x - p.R) + d_right
    else
        return p.beta * π * sin(π * x)
    end
end
# ==============================================================================
# CONVEXITY ANALYSIS TOOLS
# ==============================================================================
"""
    compute_convexity_constants(potential; verbose=true) -> ConvexityParams
Computes the convexity parameters alpha, beta, and an effective R.
For AsymmetricPotential, uses analytical expressions.
For other potentials, alpha and beta are direct, R is approximated as the point where V''(R) is close to the asymptotic value.
"""
function compute_convexity_constants(potential::AsymmetricPotential; verbose::Bool=true)
    alpha = min(potential.alpha_left, potential.alpha_right)
    R = potential.R
    beta_inside = abs(potential.beta) * π^2
    beta = max(potential.alpha_left, potential.alpha_right, beta_inside)
    if verbose
        @printf("Convexity Analysis (Asymmetric):\n")
        @printf(" - Smoothness beta: %.4f\n", beta)
        @printf(" - Convexity alpha: %.4f\n", alpha)
        @printf(" - Radius R: %.4f\n", R)
    end
    return ConvexityParams(alpha, beta, R)
end
function compute_convexity_constants(potential::Union{LocalWeakPotential, GlobalWeakPotential}; verbose::Bool=true)
    alpha = potential.alpha
    beta = potential.beta
    # Approximate R as the point where sech^2(R) ≈ 0.01, solving 4 exp(-2R) ≈ 0.01 => R ≈ (1/2) ln(400) ≈ 3.0
    R_approx = 3.0  # Conservative estimate for effective radius of transition.
    if verbose
        @printf("Convexity Analysis (%s):\n", nameof(typeof(potential)))
        @printf(" - Smoothness beta: %.4f\n", beta)
        @printf(" - Convexity alpha: %.4f\n", alpha)
        @printf(" - Effective Radius R ≈ %.4f\n", R_approx)
    end
    return ConvexityParams(alpha, beta, R_approx)
end
# ==============================================================================
# TRUE DISTRIBUTION COMPUTATION
# ==============================================================================
"""
    TrueDistribution
Holds the true stationary distribution π(x) ∝ exp(-V(x)).
Includes normalization, moments, PDF, and quantile function for Wasserstein calculations.
"""
struct TrueDistribution
    potential::AbstractPotential
    Z::Float64
    mean::Float64
    variance::Float64
    pdf::Function
    quantile_func::Function
end
"""
    compute_true_distribution(potential; rtol=1e-8, grid_size=20000) -> TrueDistribution
Numerically computes the true distribution using high-precision quadrature.
Relaxed rtol to 1e-8 for numerical stability with small alpha (wide distributions).
The grid is adaptively large to capture tails.
"""
function compute_true_distribution(potential::T; rtol::Float64=1e-8, grid_size::Int=20000) where {T<:AbstractPotential}
    integrand_Z(x) = exp(-potential(x))
    Z, _ = quadgk(integrand_Z, -Inf, Inf, rtol=rtol)
    pdf = x -> exp(-potential(x)) / Z
    integrand_mean(x) = x * pdf(x)
    mean_val, _ = quadgk(integrand_mean, -Inf, Inf, rtol=rtol)
    integrand_var(x) = (x - mean_val)^2 * pdf(x)
    var_val, _ = quadgk(integrand_var, -Inf, Inf, rtol=rtol)
    grid_range = max(30.0, 10 * sqrt(var_val))
    grid = range(mean_val - grid_range, mean_val + grid_range, length=grid_size)
    cdf_values = zeros(grid_size)
    cdf_values[1], _ = quadgk(pdf, -Inf, grid[1], rtol=1e-8)
    Threads.@threads for i in 2:grid_size
        integral, _ = quadgk(pdf, grid[i-1], grid[i], rtol=1e-8)
        cdf_values[i] = cdf_values[i-1] + integral
    end
    cdf_values ./= cdf_values[end]
    for i in 2:grid_size
        cdf_values[i] = max(cdf_values[i], cdf_values[i-1])
    end
    unique_mask = [true; diff(cdf_values) .> 1e-14]
    unique_cdf = cdf_values[unique_mask]
    unique_grid = grid[unique_mask]
    if length(unique_cdf) < 2
        @warn "Invalid CDF. Returning constant quantile."
        quantile_func = q -> mean_val
    else
        itp = linear_interpolation(unique_cdf, unique_grid, extrapolation_bc=Line())
        quantile_func = q -> itp(clamp(q, 0.0, 1.0))
    end
    return TrueDistribution(potential, Z, mean_val, var_val, pdf, quantile_func)
end
# ==============================================================================
# KERNEL DENSITY ESTIMATION FOR KL DIVERGENCE
# ==============================================================================
"""
    kde_density(samples, x; h=nothing) -> Float64
Computes the Gaussian kernel density estimate at x using Silverman's rule for bandwidth if h is not provided.
This is used for approximating the empirical density for KL calculation.
"""
function kde_density(samples::Vector{Float64}, x::Float64; h::Union{Float64, Nothing}=nothing)
    n = length(samples)
    if isnothing(h)
        std_s = std(samples)
        iqr = quantile(samples, 0.75) - quantile(samples, 0.25)
        h = 0.9 * min(std_s, iqr / 1.34) * n^(-1/5)
    end
    (1 / (n * h * sqrt(2 * π))) * sum(exp(-((x - s)^2) / (2 * h^2)) for s in samples)
end
"""
    compute_kl_divergence(true_dist, samples) -> Float64
Approximates KL(empirical || true) using KDE and numerical quadrature over a wide range.
The range is chosen to cover both empirical and true supports adequately.
"""
function compute_kl_divergence(true_dist::TrueDistribution, samples::Vector{Float64})
    n = length(samples)
    std_s = std(samples)
    iqr = quantile(samples, 0.75) - quantile(samples, 0.25)
    h = 0.9 * min(std_s, iqr / 1.34) * n^(-1/5)
    kde_func = y -> kde_density(samples, y; h=h)
    range_min = min(minimum(samples), true_dist.mean - 6 * sqrt(true_dist.variance))
    range_max = max(maximum(samples), true_dist.mean + 6 * sqrt(true_dist.variance))
    integrand = function (x)
        p = kde_func(x)
        q = true_dist.pdf(x)
        if p > 1e-15 && q > 1e-15
            p * log(p / q)
        else
            0.0
        end
    end
    kl, _ = quadgk(integrand, range_min, range_max, rtol=1e-3)
    return kl
end
# ==============================================================================
# ULA SIMULATION AND BIAS COMPUTATION
# ==============================================================================
"""
    ULAParams
Hyperparameters for ULA: step size h, number of chains, total iterations, burn-in.
"""
struct ULAParams
    h::Float64
    n_chains::Int
    n_iter::Int
    n_burn::Int
end
"""
    BiasMetrics
Comprehensive metrics for bias and diagnostics, now including KL divergence.
"""
struct BiasMetrics
    W1_bias::Float64
    W2_bias::Float64
    mean_bias::Float64
    var_bias::Float64
    kl_bias::Float64
    r_hat::Float64
    ess::Float64
end
"""
    run_ula_chains(potential, params; rng_seed=1234) -> Vector{Vector{Float64}}
Runs multiple ULA chains in parallel, collecting post-burn-in samples.
Initial positions are drawn from a wide normal to test robustness.
"""
function run_ula_chains(potential::T, params::ULAParams; rng_seed::Int=1234) where {T<:AbstractPotential}
    n_samples = params.n_iter - params.n_burn
    chains = [zeros(n_samples) for _ in 1:params.n_chains]
    sqrt_2h = sqrt(2.0 * params.h)
    x0_dist = Normal(0.0, 5.0)
    initial_positions = rand(MersenneTwister(rng_seed), x0_dist, params.n_chains)
    Threads.@threads for i in 1:params.n_chains
        local_rng = MersenneTwister(rng_seed + i)
        x = initial_positions[i]
        for _ in 1:params.n_burn
            x = x - params.h * grad(potential, x) + sqrt_2h * randn(local_rng)
        end
        for k in 1:n_samples
            x = x - params.h * grad(potential, x) + sqrt_2h * randn(local_rng)
            chains[i][k] = x
        end
    end
    return chains
end
"""
    compute_effective_sample_size(samples) -> Float64
Estimates ESS using Geyer's initial positive sequence method for autocorrelation.
"""
function compute_effective_sample_size(samples::Vector{Float64})
    n = length(samples)
    max_lag = min(1000, n ÷ 10)
    if max_lag < 2
        return Float64(n)
    end
    acf_vals = autocor(samples, 0:max_lag)
    tau_sum = 0.0
    for i in 2:2:max_lag
        term = acf_vals[i-1] + acf_vals[i]
        if term < 0
            break
        end
        tau_sum += term
    end
    tau = max(1.0, -1.0 + 2.0 * tau_sum)
    return n / tau
end
"""
    compute_rhat(chains) -> Float64
Gelman-Rubin diagnostic for chain convergence.
Values close to 1 indicate good mixing.
"""
function compute_rhat(chains::Vector{Vector{Float64}})
    M = length(chains)
    n = length(chains[1])
    chain_means = [mean(c) for c in chains]
    chain_vars = [var(c) for c in chains]
    overall_mean = mean(chain_means)
    B = n / (M - 1) * sum((chain_means .- overall_mean).^2)
    W = mean(chain_vars)
    if W < 1e-9; return 1.0; end
    var_plus = ((n-1)/n) * W + (1/M) * B
    return sqrt(var_plus / W)
end
"""
    compute_wasserstein_distance(true_dist, samples) -> (W1, W2)
Computes 1D Wasserstein distances using sorted samples and true quantiles.
"""
function compute_wasserstein_distance(true_dist::TrueDistribution, samples::Vector{Float64})
    sorted_samples = sort(samples)
    n = length(sorted_samples)
    quantiles_grid = (1:n) ./ (n + 1)
    true_quantiles = true_dist.quantile_func.(quantiles_grid)
    W1 = mean(abs.(sorted_samples .- true_quantiles))
    W2 = sqrt(mean((sorted_samples .- true_quantiles).^2))
    return W1, W2
end
"""
    compute_bias_metrics(true_dist, chains) -> BiasMetrics
Computes all bias and diagnostic metrics, including the newly added KL divergence.
"""
function compute_bias_metrics(true_dist::TrueDistribution, chains::Vector{Vector{Float64}})
    all_samples = vcat(chains...)
    W1, W2 = compute_wasserstein_distance(true_dist, all_samples)
    empirical_mean = mean(all_samples)
    empirical_var = var(all_samples)
    mean_bias = abs(empirical_mean - true_dist.mean)
    var_bias = abs(empirical_var - true_dist.variance)
    kl_bias = compute_kl_divergence(true_dist, all_samples)
    r_hat = compute_rhat(chains)
    ess = compute_effective_sample_size(all_samples)
    return BiasMetrics(W1, W2, mean_bias, var_bias, kl_bias, r_hat, ess)
end
# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================
"""
    AnalysisResult
Stores results of a single run, now including KL in metrics.
"""
struct AnalysisResult
    parameter_name::String
    parameter_value::Float64
    convexity_params::ConvexityParams
    ula_params::ULAParams
    bias_metrics::BiasMetrics
    computation_time::Float64
end
"""
    run_full_analysis(potential, ula_params, param_name, param_value; verbose=true) -> AnalysisResult
Full pipeline: convexity constants, true distribution, ULA simulation, metrics.
"""
function run_full_analysis(potential::T, ula_params::ULAParams, param_name::String, param_value::Float64; verbose::Bool=true) where {T<:AbstractPotential}
    start_time = time()
    if verbose; println("-"^60); @printf("Running Analysis for %s = %.3f\n", param_name, param_value); end
    convexity = compute_convexity_constants(potential; verbose=verbose)
    true_dist = compute_true_distribution(potential)
    if verbose; @printf("True Distribution: mean=%.4f, var=%.4f\n", true_dist.mean, true_dist.variance); end
    chains = run_ula_chains(potential, ula_params)
    metrics = compute_bias_metrics(true_dist, chains)
    comp_time = time() - start_time
    if verbose
        @printf("Bias Metrics:\n")
        @printf(" - W1 Bias: %.6f\n", metrics.W1_bias)
        @printf(" - W2 Bias: %.6f\n", metrics.W2_bias)
        @printf(" - KL Bias: %.6f\n", metrics.kl_bias)
        @printf(" - R-hat: %.4f\n", metrics.r_hat)
        @printf(" - ESS: %.1f\n", metrics.ess)
        @printf("Computation time: %.2f seconds\n", comp_time)
        println("-"^60)
    end
    return AnalysisResult(param_name, param_value, convexity, ula_params, metrics, comp_time)
end
# ==============================================================================
# CONVERGENCE RATE ANALYSIS
# ==============================================================================
"""
    ConvergenceResult
Holds W2 distances at specified time points for convergence analysis.
"""
struct ConvergenceResult
    time_points::Vector{Int}
    w2_distances::Vector{Float64}
end
"""
    analyze_convergence_rate(potential, h, max_iter, n_chains; n_points=30) -> ConvergenceResult
Runs many short chains to estimate W2 distance to stationary as a function of iterations.
Time points are log-spaced for capturing both fast and slow convergence.
"""
function analyze_convergence_rate(potential::T, h::Float64, max_iter::Int, n_chains::Int; n_points::Int=30) where {T<:AbstractPotential}
    time_points_float = exp.(range(log(10), log(max_iter), length=n_points))
    time_points = unique(sort!(round.(Int, time_points_float)))
    sqrt_2h = sqrt(2.0 * h)
    current_positions = rand(Normal(0.0, 5.0), n_chains)
    true_dist = compute_true_distribution(potential)
    w2_hist = Float64[]
    t_idx = 1
    for t in 1:max_iter
        current_positions .-= h .* grad.(Ref(potential), current_positions) .+ sqrt_2h .* randn(n_chains)
        if t == time_points[t_idx]
            _, w2 = compute_wasserstein_distance(true_dist, current_positions)
            push!(w2_hist, w2)
            t_idx += 1
            if t_idx > length(time_points)
                break
            end
        end
    end
    return ConvergenceResult(time_points[1:(t_idx-1)], w2_hist)
end
# ==============================================================================
# EXPERIMENTAL DESIGN - ALPHA DEPENDENCE ANALYSIS
# ==============================================================================
"""
    analyze_alpha_dependence(ula_params, conv_params; alpha_values, beta_val, potential_type, verbose=false)
Analyzes bias dependence on alpha for a given potential family.
Supports :local_weak, :global_weak, :asymmetric (with fixed R=2.0, beta=0.1 for convexity comparison).
"""
function analyze_alpha_dependence(ula_params::ULAParams, conv_params::@NamedTuple{R::Float64, beta_nonconv::Float64};
                                  alpha_values::Vector{Float64}, beta_val::Float64, potential_type::Symbol, verbose::Bool=false)
    type_str = string(potential_type)
    println("\n" * "="^70)
    println("EXPERIMENT: DEPENDENCE ON CONVEXITY PARAMETER ALPHA ($type_str)")
    println(@sprintf("Fixing beta=%.2f", beta_val))
    println("="^70)
    results = AnalysisResult[]
    @showprogress "Analyzing alpha dependence..." for alpha in alpha_values
        if potential_type == :local_weak
            potential = LocalWeakPotential(alpha, beta_val)
        elseif potential_type == :global_weak
            potential = GlobalWeakPotential(alpha, beta_val)
        elseif potential_type == :asymmetric
            potential = AsymmetricPotential(conv_params.R, alpha, alpha, conv_params.beta_nonconv)
        else
            error("Unknown potential_type: $potential_type")
        end
        result = run_full_analysis(potential, ula_params, "alpha", alpha; verbose=verbose)
        push!(results, result)
    end
    df = DataFrame(
        alpha = [r.parameter_value for r in results],
        W1_Bias = [r.bias_metrics.W1_bias for r in results],
        W2_Bias = [r.bias_metrics.W2_bias for r in results],
        KL_Bias = [r.bias_metrics.kl_bias for r in results],
        R_hat = [r.bias_metrics.r_hat for r in results],
        ESS = [r.bias_metrics.ess for r in results]
    )
    CSV.write("ula_bias_results/detailed_analysis/alpha_dependence_$(type_str).csv", df)
    @save "ula_bias_results/detailed_analysis/alpha_dependence_$(type_str)_results.jld2" results df
    # Dependence analysis: Check if W2 ~ 1/alpha, KL independent.
    fit_W2 = linregress(1 ./ df.alpha, df.W2_Bias)
    fit_KL = linregress(1 ./ df.alpha, df.KL_Bias)
    @printf("\nDependence Analysis (%s):\n", type_str)
    @printf(" - W2 bias vs 1/alpha slope: %.3f (R²=%.3f)\n", fit_W2.slope, fit_W2.r_squared)
    @printf(" - KL bias vs 1/alpha slope: %.3f (R²=%.3f)\n", fit_KL.slope, fit_KL.r_squared)
    # Plotting
    p1 = plot(df.alpha, [df.W2_Bias df.KL_Bias],
              xlabel="Convexity alpha", ylabel="Bias",
              title="Bias vs Alpha ($type_str)",
              marker=:circle, linewidth=2,
              label=["W2 Bias" "KL Bias"],
              xscale=:log10, yscale=:log10, legend=:topright)
    savefig("ula_bias_results/alpha_dependence_$(type_str).png")
    return results, df
end
"""
    visualize_potential_and_samples(potential, ula_params)
Visualizes the potential, true PDF, ULA histogram, and CDF comparison for qualitative assessment.
"""
function visualize_potential_and_samples(potential::AbstractPotential, ula_params::ULAParams)
    name = nameof(typeof(potential))
    println("\n" * "="^70)
    println("QUALITATIVE ANALYSIS: Potential and Samples for $name")
    println("="^70)
    true_dist = compute_true_distribution(potential)
    chains = run_ula_chains(potential, ula_params)
    all_samples = vcat(chains...)
    x_range = range(true_dist.mean - 5*sqrt(true_dist.variance), true_dist.mean + 5*sqrt(true_dist.variance), length=400)
    potential_vals = potential.(x_range)
    potential_vals .-= minimum(potential_vals)
    p1 = histogram(all_samples, normalize=:pdf, label="ULA Samples", bins=100, alpha=0.7,
                   xlabel="x", ylabel="Density", title="Distribution Comparison")
    plot!(p1, x_range, true_dist.pdf.(x_range), linewidth=3, label="True PDF")
    plot!(twinx(), x_range, potential_vals, color=:gray, linestyle=:dash, linewidth=2,
          label="Potential V(x) (shifted)", ylabel="Potential")
    sorted_samples = sort(all_samples)
    n = length(sorted_samples)
    quantiles_grid = (1:n) ./ (n + 1)
    true_quantiles = true_dist.quantile_func.(quantiles_grid)
    p2 = plot(sorted_samples, quantiles_grid, label="Empirical CDF", linewidth=2,
              title="CDF Comparison", xlabel="x", ylabel="Probability")
    plot!(true_quantiles, quantiles_grid, label="True CDF", linewidth=2, linestyle=:dash)
    plot(p1, p2, layout=(2,1), size=(800, 800), legend=:topleft)
    savefig("ula_bias_results/visualizations/qualitative_$(name).png")
    println("Visualization saved to ula_bias_results/visualizations/qualitative_$(name).png")
    return true_dist, chains
end
end # module ConvexityAnalysis
# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    # ==========================================================================
    # GLOBAL HYPERPARAMETERS (Small step size for accuracy, long runs acceptable)
    # ==========================================================================
    H_STEP_SIZE = 1e-4  # Small h for precise bias control
    H_NUM_CHAINS = 20   # Sufficient chains for reliable diagnostics
    H_NUM_ITERATIONS = 100000  # Long iterations for stationary regime
    H_BURN_IN = 10000  # Substantial burn-in
    # Convergence analysis parameters (many chains, shorter iterations)
    CONV_N_CHAINS = 50
    CONV_MAX_ITER = 100000
    # Experiment configurations
    ALPHA_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]  # Focus on small alpha to test limits
    BETA_FIXED = 2.0
    ASYM_R_FIXED = 2.0
    ASYM_BETA_NONCONV = 0.1  # Small for mild non-convexity
    base_ula_params = ConvexityAnalysis.ULAParams(H_STEP_SIZE, H_NUM_CHAINS, H_NUM_ITERATIONS, H_BURN_IN)
    conv_params = (R=ASYM_R_FIXED, beta_nonconv=ASYM_BETA_NONCONV)
    println("="^70)
    println("EMPIRICAL STUDY OF ULA BIAS AND CONVERGENCE W.R.T. CONVEXITY")
    println("Testing theoretical predictions on bias explosion in W2 vs stability in KL.")
    println("Results stored in 'ula_bias_results/'.")
    println("="^70)
    # Run alpha dependence for each family
    println("\n>>> Alpha Dependence: Local Weak Convexity")
    results_local, df_local = ConvexityAnalysis.analyze_alpha_dependence(base_ula_params, conv_params;
                                                                         alpha_values=ALPHA_VALUES, beta_val=BETA_FIXED, potential_type=:local_weak)
    println("\n>>> Alpha Dependence: Global Weak Convexity")
    results_global, df_global = ConvexityAnalysis.analyze_alpha_dependence(base_ula_params, conv_params;
                                                                           alpha_values=ALPHA_VALUES, beta_val=BETA_FIXED, potential_type=:global_weak)
    println("\n>>> Alpha Dependence: Locally Non-Convex")
    results_asym, df_asym = ConvexityAnalysis.analyze_alpha_dependence(base_ula_params, conv_params;
                                                                       alpha_values=ALPHA_VALUES, beta_val=BETA_FIXED, potential_type=:asymmetric)
    # Convergence rate analysis for select alpha (small and large)
    conv_alpha_test = [0.01, 1.0]
    for potential_type in [:local_weak, :global_weak, :asymmetric]
        type_str = string(potential_type)
        conv_results = Dict{Float64, ConvexityAnalysis.ConvergenceResult}()
        for alpha in conv_alpha_test
            if potential_type == :local_weak
                potential = ConvexityAnalysis.LocalWeakPotential(alpha, BETA_FIXED)
            elseif potential_type == :global_weak
                potential = ConvexityAnalysis.GlobalWeakPotential(alpha, BETA_FIXED)
            else
                potential = ConvexityAnalysis.AsymmetricPotential(ASYM_R_FIXED, alpha, alpha, ASYM_BETA_NONCONV)
            end
            conv_res = ConvexityAnalysis.analyze_convergence_rate(potential, H_STEP_SIZE, CONV_MAX_ITER, CONV_N_CHAINS)
            conv_results[alpha] = conv_res
        end
        # Plot convergence
        p_conv = plot(title="Convergence Rate ($type_str)", xlabel="Iterations (log)", ylabel="W2 Distance (log)", xscale=:log10, yscale=:log10)
        for (alpha, res) in conv_results
            plot!(p_conv, res.time_points, res.w2_distances, marker=:circle, label="alpha=$alpha")
            # Fit polynomial rate (expect ~1/sqrt(N) for polynomial)
            fit_poly = linregress(log.(res.time_points), log.(res.w2_distances))
            @printf("Polynomial fit slope for alpha=%.2f: %.3f\n", alpha, fit_poly.slope)
        end
        savefig(p_conv, "ula_bias_results/convergence_$(type_str).png")
    end
    # Qualitative visualization for representative cases (alpha=0.01, beta=2.0)
    println("\n>>> Qualitative Visualizations")
    local_pot = ConvexityAnalysis.LocalWeakPotential(0.01, BETA_FIXED)
    ConvexityAnalysis.visualize_potential_and_samples(local_pot, base_ula_params)
    global_pot = ConvexityAnalysis.GlobalWeakPotential(0.01, BETA_FIXED)
    ConvexityAnalysis.visualize_potential_and_samples(global_pot, base_ula_params)
    asym_pot = ConvexityAnalysis.AsymmetricPotential(ASYM_R_FIXED, 0.01, 0.01, ASYM_BETA_NONCONV)
    ConvexityAnalysis.visualize_potential_and_samples(asym_pot, base_ula_params)
    println("\n" * "="^70)
    println("ANALYSIS COMPLETE.")
    println("Examine results for bias independence in KL and potential explosion in W2 as alpha → 0.")
    println("Convergence plots show polynomial vs exponential behavior based on local/global weakness.")
    println("="^70)
end