import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Out + offsets, tl.load(X + offsets, offsets < N, 0) + tl.load(Y + offsets, offsets < N, 0), offsets < N)

@triton.jit
def affine_kernel(Scale, Shift, X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0)
    tl.store(Out + offsets, values * tl.load(Scale) + tl.load(Shift), offsets < N)

@triton.jit
def relu_product_kernel(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.maximum(tl.load(X + offsets, offsets < N, 0), 0.0) * tl.load(Y + offsets, offsets < N, 0)
    tl.store(Out + offsets, value, offsets < N)

@triton.jit
def sum_kernel(X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(N, BLOCK)):
        at = start * BLOCK + offsets
        accum += tl.load(X + at, at < N, 0)
    tl.store(Out, tl.sum(accum, 0))

@triton.jit
def mean_kernel(X, Count, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(N, BLOCK)):
        at = start * BLOCK + offsets
        accum += tl.load(X + at, at < N, 0)
    tl.store(Out, tl.sum(accum, 0) / tl.load(Count))

@triton.jit
def row_sum_kernel(X, Out, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + row * COLS + offsets, offsets < COLS, 0)
    tl.store(Out + row, tl.sum(value, 0))

@triton.jit
def rms_kernel(Count, Eps, X, Weight, Out, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + row * COLS + offsets, offsets < COLS, 0)
    scale = tl.rsqrt(tl.sum(value * value, 0) / tl.load(Count) + tl.load(Eps))
    result = value * scale * tl.load(Weight + offsets, offsets < COLS, 0)
    tl.store(Out + row * COLS + offsets, result, offsets < COLS)

@triton.jit
def softmax_kernel(X, Out, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + row * COLS + offsets, offsets < COLS, float('-inf'))
    values = tl.exp(value - tl.max(value, 0))
    result = values / tl.sum(values, 0)
    tl.store(Out + row * COLS + offsets, result, offsets < COLS)

@triton.jit
def l2_kernel(X, Out, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + row * COLS + offsets, offsets < COLS, 0)
    result = value * tl.rsqrt(tl.sum(value * value, 0))
    tl.store(Out + row * COLS + offsets, result, offsets < COLS)

@triton.jit
def matmul_kernel(B, A, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    block = tl.program_id(0)
    row = (block // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    col = (block % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        index = base * BK + inner
        a = tl.load(A + row[:, None] * K + index[None, :], (row[:, None] < M) & (index[None, :] < K), 0)
        b = tl.load(B + index[:, None] * N + col[None, :], (index[:, None] < K) & (col[None, :] < N), 0)
        accum = tl.dot(a, b, accum, input_precision='bf16x3')
    tl.store(Out + row[:, None] * N + col[None, :], accum, (row[:, None] < M) & (col[None, :] < N))

@triton.jit
def attention_kernel(Scale, Q, KT, V, Out, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    key = tl.arange(0, N)
    q = tl.load(Q + row[:, None] * D + dim[None, :], row[:, None] < M, 0)
    kt = tl.load(KT + dim[:, None] * N + key[None, :])
    score = tl.dot(q, kt, input_precision='bf16x3') / tl.load(Scale)
    p = tl.exp(score - tl.max(score, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    v = tl.load(V + key[:, None] * D + dim[None, :])
    result = tl.dot(p, v, input_precision='bf16x3')
    tl.store(Out + row[:, None] * D + dim[None, :], result, row[:, None] < M)
