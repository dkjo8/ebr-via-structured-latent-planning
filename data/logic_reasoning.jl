module LogicReasoning

using Random

export LogicProblem, LogicDataset, generate_logic_dataset,
       problem_to_tensors, check_assignment, count_satisfied

# ── CNF Representation ───────────────────────────────────────

"""
A literal is a signed integer: positive = variable, negative = negation.
A clause is a disjunction (OR) of literals.
A CNF formula is a conjunction (AND) of clauses.
"""
struct CNFFormula
    n_vars::Int
    clauses::Vector{Vector{Int}}
end

"""
Check if assignment satisfies a single clause.
Assignment is a BitVector of length n_vars (true=1, false=0).
"""
function satisfies_clause(clause::Vector{Int}, assignment::BitVector)
    for lit in clause
        var = abs(lit)
        val = assignment[var]
        satisfied = lit > 0 ? val : !val
        satisfied && return true
    end
    false
end

function count_satisfied(formula::CNFFormula, assignment::BitVector)
    count(c -> satisfies_clause(c, assignment), formula.clauses)
end

function is_satisfied(formula::CNFFormula, assignment::BitVector)
    count_satisfied(formula, assignment) == length(formula.clauses)
end

# ── Problem Definition ───────────────────────────────────────

struct LogicProblem
    formula::CNFFormula
    solution::BitVector
    is_satisfiable::Bool
    n_clauses::Int
end

struct LogicDataset
    problems::Vector{LogicProblem}
end

Base.length(ds::LogicDataset) = length(ds.problems)
Base.getindex(ds::LogicDataset, i::Int) = ds.problems[i]

# ── Generation ───────────────────────────────────────────────

"""
Generate a random k-SAT clause with `k` literals over `n_vars` variables.
"""
function random_clause(rng::AbstractRNG, n_vars::Int; k::Int=3)
    vars = sort(randperm(rng, n_vars)[1:min(k, n_vars)])
    [rand(rng, Bool) ? v : -v for v in vars]
end

"""
Generate a satisfiable CNF by planting a solution.
Creates clauses that the planted assignment satisfies,
then optionally adds a few random clauses.
"""
function generate_satisfiable(rng::AbstractRNG; n_vars::Int=5, n_clauses::Int=8, k::Int=3)
    assignment = BitVector(rand(rng, Bool, n_vars))
    clauses = Vector{Int}[]

    for _ in 1:n_clauses
        vars = sort(randperm(rng, n_vars)[1:min(k, n_vars)])
        clause = Int[]
        guaranteed = rand(rng, 1:length(vars))
        for (idx, v) in enumerate(vars)
            if idx == guaranteed
                push!(clause, assignment[v] ? v : -v)
            else
                push!(clause, rand(rng, Bool) ? v : -v)
            end
        end
        push!(clauses, clause)
    end

    formula = CNFFormula(n_vars, clauses)
    @assert is_satisfied(formula, assignment)
    LogicProblem(formula, assignment, true, n_clauses)
end

"""
Brute-force solver for small instances. Returns the first satisfying
assignment, or nothing.
"""
function brute_force_solve(formula::CNFFormula)
    n = formula.n_vars
    n > 20 && error("Brute force only for n ≤ 20")
    for i in 0:(2^n - 1)
        assignment = BitVector(digits(Bool, i; base=2, pad=n))
        is_satisfied(formula, assignment) && return assignment
    end
    nothing
end

function generate_logic_dataset(;
    n::Int=1000,
    n_vars::Int=5,
    n_clauses_range::Tuple{Int,Int}=(3, 10),
    seed::Int=42,
)
    rng = MersenneTwister(seed)
    problems = LogicProblem[]
    for _ in 1:n
        nc = rand(rng, n_clauses_range[1]:n_clauses_range[2])
        prob = generate_satisfiable(rng; n_vars, n_clauses=nc)
        push!(problems, prob)
    end
    LogicDataset(problems)
end

# ── Tensor Encoding ──────────────────────────────────────────

"""
Encode a CNF formula as a fixed-size matrix.
Rows = clauses (padded to max_clauses), columns = variables.
Values: +1 (positive literal), -1 (negative literal), 0 (absent).
"""
function problem_to_tensors(prob::LogicProblem; max_clauses::Int=16, max_vars::Int=10)
    nv = prob.formula.n_vars
    nc = prob.n_clauses

    clause_matrix = zeros(Float32, max_clauses, max_vars)
    for (ci, clause) in enumerate(prob.formula.clauses)
        ci > max_clauses && break
        for lit in clause
            var = abs(lit)
            var > max_vars && continue
            clause_matrix[ci, var] = lit > 0 ? 1f0 : -1f0
        end
    end

    target = zeros(Float32, max_vars)
    for (i, v) in enumerate(prob.solution)
        i > max_vars && break
        target[i] = v ? 1f0 : 0f0
    end

    (; clause_matrix, target, n_vars=min(nv, max_vars), n_clauses=min(nc, max_clauses))
end

function check_assignment(prob::LogicProblem, assignment::BitVector)
    is_satisfied(prob.formula, assignment)
end

end # module
