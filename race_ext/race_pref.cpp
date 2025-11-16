// rfa_prefix_mean.cpp
#include <torch/extension.h>
#include <vector>

using at::Tensor;

static void check_float32_contig(const Tensor &t, const char *name, int expected_dim)
{
    TORCH_CHECK(t.dtype() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == expected_dim, name, " must have ", expected_dim, " dims");
}

// probsK_flat: [NS,T], V_flat:[NS,T,D] -> E_flat:[NS,T,D]
// -----------------------------------------------------------------------------
// Computes causal prefix means for RACE Attention
// in a "flat" layout: each attention stream is independent.
//
// Given:
//   probsK_flat ∈ ℝ[NS, T]      -- nonnegative weights w_t per time step
//   V_flat      ∈ ℝ[NS, T, D]   -- value vectors v_t per time step
//
// For each stream s ∈ [0..NS), we compute for every time t:
//   A_t = Σ_{τ=0..t} w_τ                                  (scalar prefix sum)
//   B_t = Σ_{τ=0..t} w_τ * v_τ ∈ ℝ^D                      (vector prefix sum)
//   E_t = B_t / (A_t + eps) ∈ ℝ^D                         (prefix mean)
// and write E ∈ ℝ[NS, T, D].
//
// Notes:
// - All tensors must be CPU, float32, contiguous.
// - Accumulations are done in float64 for numerical stability.
// - Parallelization: streams (ns) are fully independent → #pragma omp parallel for.
// - Complexity: O(NS * T * D). Usually memory-bandwidth bound.
//
// -----------------------------------------------------------------------------
Tensor race_prefix_mean_flat(Tensor probsK_flat, Tensor V_flat, double eps)
{
    // ---- Shape & device checks ------------------------------------------------
    check_float32_contig(probsK_flat, "probsK_flat", 2);
    check_float32_contig(V_flat, "V_flat", 3);
    TORCH_CHECK(probsK_flat.device() == V_flat.device(), "inputs must be on same device");
    TORCH_CHECK(probsK_flat.size(0) == V_flat.size(0) && probsK_flat.size(1) == V_flat.size(1),
                "NS and T must match between probsK_flat and V_flat");

    // Dimensions:
    const int64_t NS = probsK_flat.size(0); // The M*B*H*L*R
    const int64_t T = probsK_flat.size(1);  // Tokens
    const int64_t D = V_flat.size(2);       // Embedding dimension

    // Allocate output E with same dtype/device/memory format as V_flat.
    auto E = at::empty({NS, T, D}, V_flat.options());

    // ---- Raw pointers for tight loops ----------------------------------------
    const float *PK = probsK_flat.data_ptr<float>(); // [NS, T]
    const float *VV = V_flat.data_ptr<float>();      // [NS, T, D]
    float *EO = E.data_ptr<float>();                 // [NS, T, D]

    // For probsK_flat: row stride is T (advance 1 stream = T elements).
    const int64_t stridePK_NS = T;
    // For V_flat / E: row stride is T*D (advance 1 stream = T*D elements).
    const int64_t strideV_NS = T * D;
    const int64_t strideE_NS = T * D;

    // Per-time step stride over the last dimension (D).
    const int64_t strideV_T = D;

#pragma omp parallel for
    for (int64_t ns = 0; ns < NS; ++ns)
    {
        // Causal accumulators for this stream:
        // A: scalar prefix sum of weights, B: vector prefix sum of weighted values.
        // Use double for better numerical stability when T is large.
        double A = 0.0;
        std::vector<double> B(D, 0.0);

        // Base pointers to the beginning of stream ns for each tensor.
        const float *pk_ns = PK + ns * stridePK_NS; // points to probsK_flat[ns, 0]
        const float *v_ns = VV + ns * strideV_NS;   // points to V_flat[ns, 0, 0]
        float *e_ns = EO + ns * strideE_NS;         // points to E[ns, 0, 0]

        // Walk time dimension causally: update A and B, then write normalized E_t.
        for (int64_t t = 0; t < T; ++t)
        {
            // Addresses for this time step:
            const float w = pk_ns[t];
            const float *v = v_ns + t * strideV_T; // V_flat[ns, t, :]
            float *e = e_ns + t * strideV_T;       // E[ns, t, :]

            A += static_cast<double>(w); // Read weight and update scalar prefix sum.

            // Update vector prefix sum B_d += w * v_d for all d.
            const double inv = 1.0 / (A + eps);
            for (int64_t d = 0; d < D; ++d)
            {
                B[d] += static_cast<double>(w) * static_cast<double>(v[d]);
                e[d] = static_cast<float>(B[d] * inv);
            }
        }
    }
    return E;
}

// BACKWARD: returns {grad_probsK_flat, grad_V_flat}
std::vector<Tensor> race_prefix_mean_flat_bw(
    Tensor probsK_flat, // [NS, T]
    Tensor V_flat,      // [NS, T, D]
    Tensor gradE_flat,  // [NS, T, D]
    double eps)
{
    check_float32_contig(probsK_flat, "probsK_flat", 2);
    check_float32_contig(V_flat, "V_flat", 3);
    check_float32_contig(gradE_flat, "gradE_flat", 3);

    TORCH_CHECK(probsK_flat.device() == V_flat.device() &&
                    probsK_flat.device() == gradE_flat.device(),
                "all tensors must be on the same device");

    TORCH_CHECK(probsK_flat.size(0) == V_flat.size(0) &&
                    probsK_flat.size(1) == V_flat.size(1) &&
                    V_flat.size(0) == gradE_flat.size(0) &&
                    V_flat.size(1) == gradE_flat.size(1) &&
                    V_flat.size(2) == gradE_flat.size(2),
                "shape mismatch among inputs");

    const int64_t NS = probsK_flat.size(0);
    const int64_t T = probsK_flat.size(1);
    const int64_t D = V_flat.size(2);

    auto gW = at::zeros_like(probsK_flat); // [NS, T]
    auto gV = at::zeros_like(V_flat);      // [NS, T, D]

    const float *W = probsK_flat.data_ptr<float>();
    const float *V = V_flat.data_ptr<float>();
    const float *GE = gradE_flat.data_ptr<float>();
    float *GW = gW.data_ptr<float>();
    float *GV = gV.data_ptr<float>();

    const int64_t strideW_NS = T;
    const int64_t strideV_NS = T * D;
    const int64_t strideV_T = D;
    const int64_t strideG_NS = T * D;
    const int64_t strideGW_NS = T;
    const int64_t strideGV_NS = T * D;

#pragma omp parallel for
    for (int64_t ns = 0; ns < NS; ++ns)
    {
        const float *w_ns = W + ns * strideW_NS;
        const float *v_ns = V + ns * strideV_NS;
        const float *ge_ns = GE + ns * strideG_NS;
        float *gw_ns = GW + ns * strideGW_NS;
        float *gv_ns = GV + ns * strideGV_NS;

        // Pass 1 (forward over t): compute invA[t] and alpha[t]
        std::vector<double> invA(T);
        std::vector<double> alpha(T);
        std::vector<double> B(D, 0.0);

        double A = 0.0;

        for (int64_t t = 0; t < T; ++t)
        {
            const double wt = static_cast<double>(w_ns[t]);
            const float *vt = v_ns + t * strideV_T;
            const float *gt = ge_ns + t * strideV_T;

            A += wt;
            for (int64_t d = 0; d < D; ++d)
            {
                B[d] += wt * static_cast<double>(vt[d]);
            }
            const double iA = 1.0 / (A + eps);
            invA[t] = iA;

            // alpha[t] = - (iA^2) * dot(gt, B)
            double dot_g_B = 0.0;
            for (int64_t d = 0; d < D; ++d)
            {
                dot_g_B += static_cast<double>(gt[d]) * B[d];
            }
            alpha[t] = -(iA * iA) * dot_g_B;
        }

        // Pass 2 (reverse over t): suffix sums and grads
        std::vector<double> beta_suf(D, 0.0);
        double alpha_suf = 0.0;

        for (int64_t t = T - 1; t >= 0; --t)
        {
            const float *vt = v_ns + t * strideV_T;
            const float *gt = ge_ns + t * strideV_T;
            float *gv = gv_ns + t * strideV_T;

            // beta_t = gt * invA[t]
            const double iA = invA[t];
            for (int64_t d = 0; d < D; ++d)
            {
                beta_suf[d] += static_cast<double>(gt[d]) * iA;
            }
            alpha_suf += alpha[t];

            // grad w_t = alpha_suf + dot(beta_suf, v_t)
            double dot_beta_v = 0.0;
            for (int64_t d = 0; d < D; ++d)
            {
                dot_beta_v += beta_suf[d] * static_cast<double>(vt[d]);
            }
            gw_ns[t] = static_cast<float>(alpha_suf + dot_beta_v);

            // grad v_t = w_t * beta_suf
            const double wt = static_cast<double>(w_ns[t]);
            for (int64_t d = 0; d < D; ++d)
            {
                gv[d] = static_cast<float>(wt * beta_suf[d]);
            }
        }
    }

    return {gW, gV};
}

static inline void check_input(const at::Tensor &t, at::ScalarType dtype, int64_t dims, const char *name)
{
    TORCH_CHECK(t.scalar_type() == dtype, name, " must be float32");
    TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dims, name, " must have ", dims, " dims");
}

at::Tensor race_causal_fused_forward(
    const at::Tensor &probsK, // [N,T,S] float32, CPU, contiguous
    const at::Tensor &probsQ, // [N,T,S] float32, CPU, contiguous
    const at::Tensor &V,      // [N,T,D] float32, CPU, contiguous
    double eps)
{
    check_input(probsK, at::kFloat, 3, "probsK");
    check_input(probsQ, at::kFloat, 3, "probsQ");
    check_input(V, at::kFloat, 3, "V");

    const int64_t N = probsK.size(0);
    const int64_t T = probsK.size(1);
    const int64_t S = probsK.size(2);
    TORCH_CHECK(probsQ.size(0) == N && probsQ.size(1) == T && probsQ.size(2) == S, "probsQ shape must match probsK [N,T,S]");
    TORCH_CHECK(V.size(0) == N && V.size(1) == T, "V must be [N,T,D] with same N,T as probsK");
    const int64_t D = V.size(2);

    auto out = at::empty({N, T, D}, probsK.options());

    const float *__restrict pk = probsK.data_ptr<float>(); // [N,T,S]
    const float *__restrict pq = probsQ.data_ptr<float>(); // [N,T,S]
    const float *__restrict vv = V.data_ptr<float>();      // [N,T,D]
    float *__restrict oo = out.data_ptr<float>();          // [N,T,D]

    const int64_t pk_n = T * S, pk_t = S;
    const int64_t pq_n = T * S, pq_t = S;
    const int64_t v_n = T * D, v_t = D;
    const int64_t o_n = T * D, o_t = D;

    // Tile size for D (SIMD-friendly, avoids L1/L2 thrash on large D)
    constexpr int64_t DTILE = 64;

#pragma omp parallel
    {
        // Per-thread state (avoid re-alloc per n)
        std::vector<double> A((size_t)S, 0.0);
        // Keep B in float to cut bandwidth in half; we still accumulate with doubles.
        std::vector<float> B((size_t)S * (size_t)D, 0.0f);
        std::vector<double> scale((size_t)S, 0.0); // pq / (A+eps)
        std::vector<double> outD((size_t)D, 0.0);  // double accumulator for o_nt

#pragma omp for schedule(static)
        for (int64_t n = 0; n < N; ++n)
        {
            // reset per-stream accumulators
            std::fill(A.begin(), A.end(), 0.0);
            std::fill(B.begin(), B.end(), 0.0f);

            const float *pk_nptr = pk + n * pk_n;
            const float *pq_nptr = pq + n * pq_n;
            const float *v_nptr = vv + n * v_n;
            float *o_nptr = oo + n * o_n;

            for (int64_t t = 0; t < T; ++t)
            {
                const float *pk_nt = pk_nptr + t * pk_t; // [S]
                const float *pq_nt = pq_nptr + t * pq_t; // [S]
                const float *v_nt = v_nptr + t * v_t;    // [D]
                float *o_nt = o_nptr + t * o_t;          // [D]

                // 1) Update prefixes B[s,:] += pk[s] * v[:]; A[s] += pk[s]
                //    (loop order keeps v_nt hot in cache; B laid out [s-major][d])
                for (int64_t s = 0; s < S; ++s)
                {
                    const double w = (double)pk_nt[s];
                    A[(size_t)s] += w;
                    float *Bd = &B[(size_t)s * (size_t)D];

                    // Tile D to help L1 reuse; inner loop vectorizes cleanly.
                    for (int64_t d0 = 0; d0 < D; d0 += DTILE)
                    {
                        const int64_t dmax = std::min<int64_t>(D, d0 + DTILE);
#pragma omp simd
                        for (int64_t d = d0; d < dmax; ++d)
                        {
                            Bd[d] += (float)(w * (double)v_nt[d]);
                        }
                    }
                }

// 2) Precompute scales: scale[s] = pq[s] / (A[s] + eps)
#pragma omp simd
                for (int64_t s = 0; s < S; ++s)
                {
                    scale[(size_t)s] = (double)pq_nt[s] / (A[(size_t)s] + eps);
                }

                // 3) out = sum_s scale[s] * B[s,:]
                //    Accumulate in double, cast once at the end.
                std::fill(outD.begin(), outD.end(), 0.0);

                for (int64_t s = 0; s < S; ++s)
                {
                    const double alpha = scale[(size_t)s];
                    const float *Bd = &B[(size_t)s * (size_t)D];

                    for (int64_t d0 = 0; d0 < D; d0 += DTILE)
                    {
                        const int64_t dmax = std::min<int64_t>(D, d0 + DTILE);
#pragma omp simd
                        for (int64_t d = d0; d < dmax; ++d)
                        {
                            outD[d] += alpha * (double)Bd[d];
                        }
                    }
                }

// 4) store once (cast)
#pragma omp simd
                for (int64_t d = 0; d < D; ++d)
                {
                    o_nt[d] = (float)outD[d];
                }
            } // t
        } // n
    } // parallel

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("race_prefix_mean_flat", &race_prefix_mean_flat, "Race prefix mean (probsK[NS,T], V[NS,T,D])");
    m.def("race_prefix_mean_flat_bw", &race_prefix_mean_flat_bw, "Race prefix mean backward (flat)");
    m.def("race_causal_fused_forward", &race_causal_fused_forward, "Causal fused forward");
}