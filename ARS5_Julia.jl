# ARS 5.0 - Julia with Flux.jl
# ============================================================================

using Flux
using Statistics
using Random

# ============================================================================
# 1. SYMBOLIC COMPONENT
# ============================================================================

struct ARSGrammar
    symbols::Vector{String}
    symbol_to_idx::Dict{String,Int}
    n_symbols::Int
    counts::Matrix{Int}
    probs::Matrix{Float64}
    constitutive_rules::Set{Tuple{Int,Int}}
end

function ARSGrammar()
    symbols = ["KBG", "VBG", "KBBd", "VBBd", "KBA", "VBA", 
               "KAE", "VAE", "KAA", "VAA", "KAV", "VAV"]
    n = length(symbols)
    symbol_to_idx = Dict(s => i-1 for (i, s) in enumerate(symbols))
    counts = zeros(Int, n, n)
    probs = ones(Float64, n, n) / n
    
    # Constitutive rules
    rules = Set{Tuple{Int,Int}}()
    push!(rules, (symbol_to_idx["KBG"], symbol_to_idx["VBG"]))
    push!(rules, (symbol_to_idx["KAV"], symbol_to_idx["VAV"]))
    push!(rules, (symbol_to_idx["VAV"], symbol_to_idx["KAV"]))
    
    return ARSGrammar(symbols, symbol_to_idx, n, counts, probs, rules)
end

function update_symbolic!(grammar::ARSGrammar, from_idx::Int, to_idx::Int)
    grammar.counts[from_idx+1, to_idx+1] += 1
    row_sum = sum(grammar.counts[from_idx+1, :])
    if row_sum > 0
        grammar.probs[from_idx+1, :] = grammar.counts[from_idx+1, :] / row_sum
    end
end

function is_valid_transition(grammar::ARSGrammar, from_idx::Int, to_idx::Int)::Bool
    return !((from_idx, to_idx) in grammar.constitutive_rules)
end

# ============================================================================
# 2. NEURAL COMPONENT
# ============================================================================

struct ARSNeuralNetwork
    model::Any
    n_symbols::Int
end

function ARSNeuralNetwork(n_symbols::Int; hidden::Int=64)
    model = Chain(
        Dense(n_symbols, hidden, relu),
        Dropout(0.2),
        Dense(hidden, hidden ÷ 2, relu),
        Dropout(0.2),
        Dense(hidden ÷ 2, n_symbols),
        softmax
    )
    return ARSNeuralNetwork(model, n_symbols)
end

function predict(neural::ARSNeuralNetwork, from_idx::Int)
    x = zeros(Float32, 1, neural.n_symbols)
    x[1, from_idx+1] = 1.0f0
    return neural.model(x)[1, :]
end

function train_step!(neural::ARSNeuralNetwork, from_idx::Int, to_idx::Int, grammar::ARSGrammar; lr=0.001f0)
    # Target: symbolic probabilities
    target = grammar.probs[from_idx+1, :]
    
    loss, grad = Flux.withgradient(neural.model) do m
        x = zeros(Float32, 1, neural.n_symbols)
        x[1, from_idx+1] = 1.0f0
        y_pred = m(x)[1, :]
        # Cross-entropy loss
        return -sum(target .* log.(y_pred .+ 1f-10))
    end
    
    Flux.update!(Flux.setup(Adam(lr), neural.model), neural.model, grad)
    return loss
end

# ============================================================================
# 3. HYBRID SYSTEM
# ============================================================================

mutable struct ARSNeuroSymbolicSystem
    grammar::ARSGrammar
    neural::ARSNeuralNetwork
    loss_history::Vector{Float64}
end

function ARSNeuroSymbolicSystem()
    grammar = ARSGrammar()
    neural = ARSNeuralNetwork(grammar.n_symbols)
    return ARSNeuroSymbolicSystem(grammar, neural, Float64[])
end

function train_transition!(system::ARSNeuroSymbolicSystem, from_sym::String, to_sym::String)
    from_idx = system.grammar.symbol_to_idx[from_sym]
    to_idx = system.grammar.symbol_to_idx[to_sym]
    
    # Symbolic update (fast)
    update_symbolic!(system.grammar, from_idx, to_idx)
    
    # Neural update (slow)
    loss = train_step!(system.neural, from_idx, to_idx, system.grammar)
    push!(system.loss_history, loss)
    
    return loss
end

function train_corpus!(system::ARSNeuroSymbolicSystem, corpus::Vector{Vector{String}}; epochs=10)
    println("Training on $(length(corpus)) transcripts for $epochs epochs...")
    
    for epoch in 1:epochs
        total_loss = 0.0
        n_transitions = 0
        
        for chain in corpus
            for i in 1:length(chain)-1
                loss = train_transition!(system, chain[i], chain[i+1])
                total_loss += loss
                n_transitions += 1
            end
        end
        
        avg_loss = total_loss / n_transitions
        println("Epoch $epoch/$epochs, Avg Loss: $(round(avg_loss, digits=6))")
    end
end

function predict_next(system::ARSNeuroSymbolicSystem, from_sym::String)::Dict{String,Float64}
    from_idx = system.grammar.symbol_to_idx[from_sym]
    
    # Neural prediction
    neural_probs = predict(system.neural, from_idx)
    
    # Symbolic probabilities
    symbolic_probs = system.grammar.probs[from_idx+1, :]
    
    # Combine
    combined = 0.5 .* neural_probs .+ 0.5 .* symbolic_probs
    
    return Dict(system.grammar.symbols[i] => combined[i] 
                for i in 1:system.grammar.n_symbols)
end

function generate_sequence(system::ARSNeuroSymbolicSystem, max_len::Int=20, start_sym::String="KBG")
    seq = [start_sym]
    
    for _ in 1:max_len-1
        probs = predict_next(system, seq[end])
        
        # Filter by constitutive rules
        from_idx = system.grammar.symbol_to_idx[seq[end]]
        valid_pairs = [(s, p) for (s, p) in probs if is_valid_transition(system.grammar, from_idx, system.grammar.symbol_to_idx[s])]
        
        if isempty(valid_pairs)
            break
        end
        
        # Normalize and sample
        valid_symbols = [p[1] for p in valid_pairs]
        valid_probs = [p[2] for p in valid_pairs]
        valid_probs ./= sum(valid_probs)
        
        next_sym = rand(valid_symbols, Weights(valid_probs))
        push!(seq, next_sym)
        
        if next_sym in ["KAV", "VAV"]
            break
        end
    end
    
    return seq
end

# ============================================================================
# 4. CORPUS AND DEMONSTRATION
# ============================================================================

corpus = [
    ["KBG", "VBG", "KBBd", "VBBd", "KBA", "VBA", "KBBd", "VBBd", "KBA", "VAA", "KAA", "VAV", "KAV"],
    ["VBG", "KBBd", "VBBd", "VAA", "KAA", "VBG", "KBBd", "VAA", "KAA"],
    ["KBBd", "VBBd", "VAA", "KAA"],
    ["KBBd", "VBBd", "KBA", "VBA", "KBBd", "VBA", "KAE", "VAE", "KAA", "VAV", "KAV"],
    ["KAV", "KBBd", "VBBd", "KBBd", "VAA", "KAV"],
    ["KBG", "VBG", "KBBd", "VBBd", "KAA"],
    ["KBBd", "VBBd", "KBA", "VAA", "KAA"],
    ["KBG", "VBBd", "KBBd", "VBA", "VAA", "KAA", "VAV", "KAV"]
]

function main()
    println("=" ^ 70)
    println("ARS 5.0 - Julia with Flux.jl")
    println("The Empirical Grammar of Market Conversations")
    println("=" ^ 70)
    
    system = ARSNeuroSymbolicSystem()
    
    println("\n--- Training ---")
    train_corpus!(system, corpus, epochs=20)
    
    println("\n--- Learned Transition Probabilities ---")
    for from_sym in ["KBG", "KBBd", "VBA", "KAA"]
        probs = predict_next(system, from_sym)
        top = sort(collect(probs), by=x->x[2], rev=true)[1:3]
        println("$from_sym → $(join(["$s: $(round(p, digits=3))" for (s, p) in top], ", "))")
    end
    
    println("\n--- Generated Sequences ---")
    for i in 1:5
        seq = generate_sequence(system)
        println("Seq $i: $(join(seq, " → "))")
    end
    
    return system
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end