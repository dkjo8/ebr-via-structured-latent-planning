module ArithmeticReasoning

using Random

export ArithmeticProblem, ArithmeticDataset, generate_arithmetic_dataset,
       problem_to_tensors, evaluate_expression

# ── Expression Tree ──────────────────────────────────────────

abstract type Expr end

struct Literal <: Expr
    value::Int
end

struct BinOp <: Expr
    op::Symbol      # :+, :-, :*
    left::Expr
    right::Expr
end

function evaluate(e::Literal)::Float64
    Float64(e.value)
end

function evaluate(e::BinOp)::Float64
    l = evaluate(e.left)
    r = evaluate(e.right)
    if e.op == :+
        l + r
    elseif e.op == :-
        l - r
    elseif e.op == :*
        l * r
    else
        error("Unknown op: $(e.op)")
    end
end

function to_string(e::Literal)::String
    string(e.value)
end

function to_string(e::BinOp)::String
    "($(to_string(e.left)) $(e.op) $(to_string(e.right)))"
end

"""
Collect all intermediate computation steps in evaluation order (post-order).
Each step is (sub_expression_string, result_value).
"""
function collect_steps(e::Literal)
    [(to_string(e), evaluate(e))]
end

function collect_steps(e::BinOp)
    steps = Tuple{String, Float64}[]
    append!(steps, collect_steps(e.left))
    append!(steps, collect_steps(e.right))
    push!(steps, (to_string(e), evaluate(e)))
    steps
end

# ── Problem Definition ───────────────────────────────────────

struct ArithmeticProblem
    expression::Expr
    expression_str::String
    answer::Float64
    steps::Vector{Tuple{String, Float64}}
end

struct ArithmeticDataset
    problems::Vector{ArithmeticProblem}
end

Base.length(ds::ArithmeticDataset) = length(ds.problems)
Base.getindex(ds::ArithmeticDataset, i::Int) = ds.problems[i]

# ── Expression Generation ────────────────────────────────────

const OPS = [:+, :-, :*]

function random_expr(rng::AbstractRNG, depth::Int; max_operand::Int=99)
    if depth <= 0 || (depth == 1 && rand(rng) < 0.5)
        return Literal(rand(rng, 1:max_operand))
    end
    op = OPS[rand(rng, 1:length(OPS))]
    left = random_expr(rng, depth - 1; max_operand)
    right = random_expr(rng, depth - 1; max_operand)
    BinOp(op, left, right)
end

function generate_problem(rng::AbstractRNG; max_depth::Int=4, max_operand::Int=99)
    depth = rand(rng, 2:max_depth)
    expr = random_expr(rng, depth; max_operand)
    answer = evaluate(expr)
    steps = collect_steps(expr)
    ArithmeticProblem(expr, to_string(expr), answer, steps)
end

function generate_arithmetic_dataset(;
    n::Int=1000,
    max_depth::Int=4,
    max_operand::Int=99,
    seed::Int=42,
)
    rng = MersenneTwister(seed)
    problems = [generate_problem(rng; max_depth, max_operand) for _ in 1:n]
    ArithmeticDataset(problems)
end

# ── Tensor Encoding ──────────────────────────────────────────

const TOKEN_VOCAB = Dict{Char, Int}(
    '(' => 1, ')' => 2, '+' => 3, '-' => 4, '*' => 5, ' ' => 6,
    '0' => 7, '1' => 8, '2' => 9, '3' => 10, '4' => 11,
    '5' => 12, '6' => 13, '7' => 14, '8' => 15, '9' => 16,
)
const VOCAB_SIZE = 16

"""
Encode expression string as a padded integer sequence.
"""
function tokenize(s::String; max_len::Int=64)
    tokens = zeros(Int, max_len)
    for (i, c) in enumerate(s)
        i > max_len && break
        tokens[i] = get(TOKEN_VOCAB, c, 0)
    end
    tokens
end

"""
Convert problem to tensor representation.
Returns token sequence and target steps encoded as Float32 vectors.
"""
function problem_to_tensors(prob::ArithmeticProblem; max_seq_len::Int=64, max_steps::Int=16)
    tokens = tokenize(prob.expression_str; max_len=max_seq_len)

    step_values = zeros(Float32, max_steps)
    for (i, (_, val)) in enumerate(prob.steps)
        i > max_steps && break
        step_values[i] = Float32(val)
    end

    (; tokens, step_values, answer=Float32(prob.answer), n_steps=min(length(prob.steps), max_steps))
end

function evaluate_expression(prob::ArithmeticProblem)
    evaluate(prob.expression)
end

end # module
