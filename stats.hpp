#pragma once
#include <cassert>
#include <vector>
#include <queue>
#include <algorithm>
#include <print>
#include "blas_lapack.hpp"
#include "random.hpp"
#include <gsl/gsl_cdf.h>

namespace stats
{

// Divide and conquer vector addition algorithms.

// Via backward iteration. This is garbage on GPU obviously. This destroys the
// input.
double sum_1
( size_t n, double *x
)
{ for (; n > 1; n = (n+1)/2)
    for (size_t i = 0; i < n/2; i++)
    { size_t L_i = i;
      size_t R_i = n-i-1;
      x[i] += x[n-i-1];
    }
  return *x;
}

// Forward iteration with jumps. This is (conceptually at least) what one should
// do on the GPU, short of all sorts of optimization obviously. This still
// destroys the input, so it requires 200% memory.
double sum_2
( size_t n, double *x
)
{ for (size_t k = 1; k < n; k *= 2)
  { for (size_t i = 0; i < (n+k)/(2*k); i++)
    { size_t L_i = k*(2*i  );
      size_t R_i = k*(2*i+1);
      if (R_i < n)
        x[L_i] += x[R_i];
    }      
  }
  return *x;
}

// Like sum_2 but does not destroy x, optimized to copy only half of x. At this
// point it is good to realize that we do pairwise sums, but this is not
// necessary. If we do sums in groups of 8 say, the errors still scale quite
// well, the algorithm would need around log_2(8)=3 times fewer layers, and the
// memory required would be 112.5% of the usual, instead of the 150% here.
double sum_3
( size_t n, const double *x, size_t offset = 0
)
{ size_t m = (n+1)/2;
  std::vector<double> y(m);
  for (size_t i = 0; i < m; i++)
  { size_t L_i = 2*i;
    size_t R_i = 2*i+1;
    y[i] = x[L_i];
    if (R_i < n)
      y[i] += x[R_i];
  }
  for (size_t k = 1; k < m; k *= 2)
  { for (size_t i = 0; i < (m+k)/(2*k); i++)
    { size_t L_i = k*(2*i  );
      size_t R_i = k*(2*i+1);
      if (R_i < m)
        y[L_i] += y[R_i];
    }      
  }
  return y[0];
}

// Like sum_3 but acts on (contingent) vectors instead of elements.
void sum_vectors
( size_t n_dims, size_t n, const double *x, double *sum
)
{ size_t m = (n+1)/2;
  std::vector<double> y(m*n_dims);
  for (size_t i = 0; i < m; i++)
  { for (size_t j = 0; j < n_dims; j++)
    { size_t L_i = (2*i  )*n_dims+j;
      size_t R_i = (2*i+1)*n_dims+j;
      size_t R_i_ = (2*i+1);
      size_t l = i*n_dims+j;
      y[l] = x[L_i];
      if (R_i_ < n)
        y[l] += x[R_i];
    }
  }
  for (size_t k = 1; k < m; k *= 2)
  { for (size_t i = 0; i < (m+k)/(2*k); i++)
    { for (size_t j = 0; j < n_dims; j++)
      { size_t L_i = k*(2*i  )*n_dims+j;
        size_t R_i = k*(2*i+1)*n_dims+j;
        size_t R_i_ = k*(2*i+1);
        if (R_i_ < m)
          y[L_i] += y[R_i];
      }
    }      
  }
  for (size_t j = 0; j < n_dims; j++)
    sum[j] = y[j];
}

// Via naive sum. sample_mean must be 0-initialized.
void sample_mean_1
( size_t n_dims, size_t n_samples, const double *samples, double *sample_mean
)
{ for (size_t i = 0; i < n_samples; i++)
    for (size_t j = 0; j < n_dims; j++)
      sample_mean[j] += samples[i*n_dims+j];
  for (size_t j = 0; j < n_dims; j++)
    sample_mean[j] /= n_samples;
}

// Via divide-and-conquer sum.
void sample_mean_2
( size_t n_dims, size_t n_samples, const double *samples, double *sample_mean
)
{ sum_vectors(n_dims, n_samples, samples, sample_mean);
  for (size_t i = 0; i < n_dims; i++)
    sample_mean[i] /= n_samples;
}

void samples_center
( size_t n_dims, size_t n_samples, double *samples, const double *sample_mean
)
{ for (size_t i = 0; i < n_samples; i++)
    for (size_t j = 0; j < n_dims; j++)
      samples[i*n_dims+j] -= sample_mean[j];
}

void samples_cen
( size_t n_dims, size_t n_samples, const double *samples, const double *sample_mean, double *samples_cen
)
{ memcpy(samples_cen, samples, sizeof(double)*n_samples*n_dims);
  samples_center(n_dims, n_samples, samples_cen, sample_mean);
}

// If not maximum_likelihood, it used the unbiased estimator instead.
template <bool maximum_likelihood = false>
void sample_cov
( size_t n_dims, size_t n_samples, const double *samples_cen, double *sample_cov
)
{ const int n_dims_ = n_dims, n_samples_ = n_samples;
  const double alpha = maximum_likelihood ? 1./n_samples : 1./(n_samples-1);
  constexpr double beta = 0;
  blas_lapack::dsyrk_("L", "N", &n_dims_, &n_samples_, &alpha, samples_cen, &n_dims_, &beta, sample_cov, &n_dims_);
}

double det_block_diag
( size_t n, const double *a, const int *ipiv
)
{ double log_det = 0;
  for (size_t i = 0; i < n;)
  { if (ipiv[i] > 0)
    { log_det += log(abs(a[i*n+i]));
      i++;
    } else
    { log_det += log(abs(a[i*n+i]*a[(i+1)*(n+1)]-pow(a[(i+1)*n+i],2)));
      i += 2;
    }
  }
  return exp(log_det);
}

// Squared Mahalanobis distance from inverse sample covariance.
void Mah_dist2_raw
( size_t n_dims, size_t n_samples, const double *samples_cen, const double *sample_mean, const double *sample_cov_inv, double *samples_dist, double *work_1, double *work_2
)
{ int n_dims_ = n_dims;
  for (size_t i = 0; i < n_samples; i++)
  { constexpr int inc = 1;
    constexpr double alpha = 1, beta = 0;
    memcpy(work_1, samples_cen+n_dims*i, sizeof(double)*n_dims);
    blas_lapack::dgemv_("N", &n_dims_, &n_dims_, &alpha, sample_cov_inv, &n_dims_, work_1, &inc, &beta, work_2, &inc);
    samples_dist[i] = blas_lapack::ddot_(&n_dims_, samples_cen+n_dims*i, &inc, work_2, &inc);
  }
}

// Squared Mahalanobis distance from sample covariance. Can also return the
// covariance determinant.
template <bool return_cov_det = false>
auto Mah_dist2
( size_t n_dims, size_t n_samples, const double *samples_cen, const double *sample_mean, const double *sample_cov,
  double *samples_dist, double *sample_cov_inv, double *work_1, double *work_2, int *ipiv
)
{ int n_dims_ = n_dims, lp_info;
  blas_lapack::dlacpy_("L", &n_dims_, &n_dims_, sample_cov, &n_dims_, sample_cov_inv, &n_dims_);
  blas_lapack::dsytrf_("L", &n_dims_, sample_cov_inv, &n_dims_, ipiv, work_1, &n_dims_, &lp_info);
  assert(!lp_info);
  double det;
  if constexpr (return_cov_det)
    det = det_block_diag(n_dims, sample_cov_inv, ipiv);
  blas_lapack::dsytri_("L", &n_dims_, sample_cov_inv, &n_dims_, ipiv, work_1, &lp_info);
  assert(!lp_info);
  Mah_dist2_raw(n_dims, n_samples, samples_cen, sample_mean, sample_cov_inv, samples_dist, work_1, work_2);
  if constexpr (return_cov_det)
    return det;
}

// Same as above, with internal buffers.
template <bool return_cov_det = false>
auto Mah_dist2
( size_t n_dims, size_t n_samples, const double *samples_cen, const double *sample_mean, const double *sample_cov, double *samples_dist
)
{ int n_dims_ = n_dims;
  std::vector<double> sample_cov_inv(pow(n_dims,2), 0),
                      work_1(n_dims, 0),
                      work_2(n_dims, 0);
  std::vector<int> ipiv(n_dims, 0);
  return Mah_dist2<return_cov_det>(n_dims, n_samples, samples_cen, sample_mean, sample_cov, samples_dist, sample_cov_inv.data(), work_1.data(), work_2.data(), ipiv.data());
}

// A max heap is a highly specific priority queue, where the number of elements
// is fixed, and with a comparison operator that cares only about the 1st
// element in a pair. In principle a standard priority queue works fine, but
// fixing the number of elements by constructing the underlying data ourselves
// and defining a custom comparison operator different from the implicit one for
// pairs are optimizations. It should be done properly.
template <typename T_priority, typename T_object>
auto create_max_heap(size_t n)
{ using Pair = std::pair<T_priority, T_object>;
// Reserve data, don't 0-initialize data, because internally it obviously uses
// push_back.
  std::vector<Pair> data;
  data.reserve(n);
  auto less = [](const Pair &a, const Pair &b)
  { return a.first < b.first;
  };
  return std::priority_queue<Pair, decltype(data), decltype(less)> {less, std::move(data)};
}

// Find the m smallest numbers in x of length n and put their indices in min,
// smallest first.
// (Ended up not using this in the MCD and doing it manually, but it is easier
// to understand it on its own as written here.)
void partial_sort_smallest
( size_t m, size_t n, const double *x, size_t *min
)
{ auto max_heap = create_max_heap<double, size_t>(m); // priority, idx
  for (size_t i = 0; i < n; ++i)
  { if (max_heap.size() < m)
      max_heap.push({x[i], i});
    else if (x[i] < max_heap.top().first)
    { max_heap.pop();
      max_heap.push({x[i], i});
    }
  }
  for (size_t i = 0; i < m; i++)
  { min[m-i-1] = max_heap.top().second;
    max_heap.pop();
  }
}

// Sample permutations via Fisher-Yates.

// In-place algorithm, permuting objects directly.
void permute_samples_1
( pcg_xsl_rr_128_64::Generator &gen, size_t n_dims, size_t n_samples, double *samples
)
{ for (size_t i = n_samples-1; i > 0; i--)
  { size_t k = gen.next()%(i+1);
    for (size_t j = 0; j < n_dims; j++)
      std::swap(samples[n_dims*i+j], samples[n_dims*k+j]);
  }
}

// Out-of-place algorithm, but permutes indices instead of objects, so fewer
// swaps, superior for large n_dims.
void permute_samples_2
( pcg_xsl_rr_128_64::Generator &gen, size_t n_dims, size_t n_samples, double *samples
)
{ std::vector<double> samples_old(n_samples*n_dims);
  memcpy(samples_old.data(), samples, sizeof(double)*n_samples*n_dims);
  std::vector<size_t> ids(n_samples);
  for (size_t i = 0; i < n_samples; i++)
    ids[i] = i;
  gen.permute(n_samples, ids.data());
  for (size_t i = 0; i < n_samples; i++)
    for (size_t j = 0; j < n_dims; j++)
      samples[n_dims*i+j] = samples_old[n_dims*ids[i]+j];
}

// Create a subset of a permutation of the samples. The subset does not have to
// be random since the permutation is already uniformly random, that's the
// 'trick'. Write the corresponding sample indices in ids_out if not null.
void permute_subset_samples
( pcg_xsl_rr_128_64::Generator &gen, size_t n_dims, size_t n_samples_in, const double *samples_in, size_t n_samples_out,
  double *samples_out, size_t *ids_out = nullptr
)
{ std::vector<size_t> ids(n_samples_in);
  for (size_t i = 0; i < n_samples_in; i++)
    ids[i] = i;
  gen.permute(n_samples_in, ids.data());
  for (size_t i = 0; i < n_samples_out; i++)
    for (size_t j = 0; j < n_dims; j++)
      samples_out[n_dims*i+j] = samples_in[n_dims*ids[i]+j];
  if (ids_out)
    for (size_t i = 0; i < n_samples_out; i++)
      ids_out[i] = ids[i];
}

// Minimum covariance determinant, with reweighting. (It's simpler to keep it
// all in one function as opposed to splitting the raw and reweighted MCD.)
template <size_t n_C0_attempts = 500, size_t n_C0_keep = 10, size_t n_C_steps = 2, size_t max_iter = 100>
void MCD
( pcg_xsl_rr_128_64::Generator &gen, size_t n_dims, size_t n_samples, const double *samples, size_t h, double *MCD_mean, double *MCD_cov
)
{
// First we implement the so-called 'C_step':
// Compute Mahalobis distances from mean and cov, take h observations with
// smallest distance, update mean and cov to be associated to said observations,
// and remember covariance determinant.
// A max heap is used to keep track of the lowest Mahalanobis distance samples.
// The mean and covariance here must be initialized by the size n_dims+1 subset
// later, the C-step initial condition as we will refer to it.
// C_subset is centered, i.e. the original data is destroyed, so the meaning
// changes.
  std::vector<double> samples_cen_(n_samples*n_dims),
                      Mah_dists2(n_samples),
                      C_subset(h*n_dims),
                      C_mean(n_dims),
                      C_cov(pow(n_dims,2)),
                      C_cov_inv(pow(n_dims,2), 0),
                      work_1(n_dims, 0),
                      work_2(n_dims, 0);
  std::vector<int> ipiv(n_dims, 0);
  double cov_det;
  auto max_heap_samples = create_max_heap<double, size_t>(h); // Mah dist, sample idx
// Raw MCD consistency factor (at the normal model, no finite sample
// correction).
  const double alpha = static_cast<double>(h)/n_samples;
  const double chi2_q_alpha = gsl_cdf_chisq_Pinv(1-alpha, n_dims);
  const double chi2_cum = gsl_cdf_chisq_P(chi2_q_alpha, n_dims+2);
// This is really quite odd, if it wasn't for the degrees of freedom being
// different, this would be a cumulative distribution function of a quantile
// (inverse cumulative distribution function), so it would be alpha/(1-alpha).
// In fact, this is the limit in a large number of dimensions.
  const double c0 = alpha/chi2_cum; // raw MCD consistency factor
  auto C_step = [&]()
  {
// Calculate Mahalanobis distances from current mean and covariance, and get
// covariance determinant. This requires first sampling the samples via the
// current mean. The consistency factor is already applied after the C-step
// initial condition (shown later).
    samples_cen(n_dims, n_samples, samples, C_mean.data(), samples_cen_.data());
    cov_det = Mah_dist2<true>(n_dims, n_samples, samples_cen_.data(), C_mean.data(), C_cov.data(), Mah_dists2.data(), C_cov_inv.data(), work_1.data(), work_2.data(), ipiv.data());
// Fill max heap of samples and empty it again, creating a contiguous subset
// of the lowest covariance determinant data. (It is reverse sorted as
// implemented here, but this is irrelevant.) This is partial_sort_smallest
// essentially, but empties samples instead of indices.
    for (size_t i = 0; i < n_samples; i++)
    { if (max_heap_samples.size() < h)
        max_heap_samples.push({Mah_dists2[i], i});
      else if (Mah_dists2[i] < max_heap_samples.top().first)
      { max_heap_samples.pop();
        max_heap_samples.push({Mah_dists2[i], i});
      }
    }
    for (size_t i = 0; i < h; i++)
    { size_t k = max_heap_samples.top().second;
      max_heap_samples.pop();
      for (size_t j = 0; j < n_dims; j++)
        C_subset[n_dims*i+j] = samples[n_dims*k+j];
    }
// Analyze said subset, i.e. update mean and covariance and apply the
// consistency factor. Here C_subset is centered and the data thus destroyed, be
// careful.
    sample_mean_2(n_dims, h, C_subset.data(), C_mean.data());
    samples_center(n_dims, h, C_subset.data(), C_mean.data());
    sample_cov(n_dims, h, C_subset.data(), C_cov.data());
    for (size_t i = 0; i < pow(n_dims,2); i++)
      C_cov[i] *= c0;
  };

// The first step in the raw MCD algorithm is to sample what we call here C_step
// initial conditions, remembering the ones leading to a low covariance
// determinant.
// We refer to a C-step initial condition as C0 here. Here C0_subset will be
// centered at some point, destroying the data; its meaning changes.
// A max heap is used to keep track of the lowest covariance determinant
// h-subsets. We don't remember these subsets, instead we remember the RNG seed
// used to generate them and just regenerate them later. Either way is fine
// really, it is not much memory nor much computation, but especially on GPU the
// computation comes cheap so we do it this way.
  std::vector<double> C0_subset((n_dims+1)*n_dims);
  auto max_heap_subsets = create_max_heap<double, pcg_xsl_rr_128_64::Generator>(n_C0_keep); // cov det, RNG state
  auto try_C0_subset = [&]()
  { 
// Sample the C0 subset (we wrote a nice function for this), and remember the
// RNG state.
    auto gen0 = gen;
    permute_subset_samples(gen, n_dims, n_samples, samples, n_dims+1, C0_subset.data());
// Compute its mean and covariance and apply the consistency factor. This
// requires centering the data. Careful, the data is centered in-place, so the
// uncentered data is destroyed.
    sample_mean_2(n_dims, n_dims+1, C0_subset.data(), C_mean.data());
    samples_center(n_dims, n_dims+1, C0_subset.data(), C_mean.data());
    sample_cov(n_dims, n_dims+1, C0_subset.data(), C_cov.data());
    for (size_t i = 0; i < pow(n_dims,2); i++)
      C_cov[i] *= c0;
// Perform the preliminary C-steps. Now the covariance determinant of the
// h-subset is known. Update the max heap using said covariance determinant.
    for (size_t i = 0; i < n_C_steps; i++)
      C_step();
    if (max_heap_subsets.size() < n_C0_keep)
      max_heap_subsets.push({cov_det, gen0});
    else if (cov_det < max_heap_subsets.top().first)
    { max_heap_subsets.pop();
      max_heap_subsets.push({cov_det, gen0});
    }
  };
  for (size_t i = 0; i < n_C0_attempts; i++)
    try_C0_subset();

// For the n_C0_keep best candidates, apply C-steps until they converge. We use
// the RNG state here to regenerate those candidates.
  double cov_det_best = std::numeric_limits<double>::max();
  pcg_xsl_rr_128_64::Generator gen_best;
  auto converge = [&]()
  { C_step();
    double cov_det_old = cov_det;
    for (size_t j = 0; j < max_iter-1; j++)
    { C_step();
      double cov_det_rel_change = (cov_det-cov_det_old)/cov_det_old;
      if (cov_det_rel_change < 1e-6)
        break;
    }
  };
// Reconstruct and converge the best candidates.
  auto reconstruct = [&]()
  { permute_subset_samples(gen, n_dims, n_samples, samples, n_dims+1, C0_subset.data());
    sample_mean_2(n_dims, n_dims+1, C0_subset.data(), C_mean.data());
    samples_center(n_dims, n_dims+1, C0_subset.data(), C_mean.data());
    sample_cov(n_dims, n_dims+1, C0_subset.data(), C_cov.data());
    for (size_t i = 0; i < pow(n_dims,2); i++)
      C_cov[i] *= c0;
  };
  for (size_t i = 0; i < n_C0_keep; i++)
  {
// Reconstruction and convergence.
    auto gen0 = max_heap_subsets.top().second;
    gen = gen0;
    max_heap_subsets.pop();
    reconstruct();
    converge();
// Remember the winner.
    if (cov_det < cov_det_best)
    { cov_det_best = cov_det;
      gen_best = gen0;
    }
  }
// Regenerate the winner, completing the 'raw MCD' algorithm.
  gen = gen_best;
  reconstruct();
  converge();

// Reweighted MCD.
  const double dist2_max = gsl_cdf_chisq_Pinv(1-0.025, n_dims); // alpha=2.5%
// Reweighted MCD consistency factor (at the normal model, no finite sample
// correction).
// (Incomplete obviously, see README.)
  const double c1 = 1;
// Find the samples accepted to compute the mean and covariance that become the
// MCD mean and covariance. We reuse the centered samples buffer since it's
// sufficiently large and no longer needed. The required Mahalanobis distances
// were already computed during regeneration of the winner candidate.
  for (size_t i = 0; i < n_samples*n_dims; i++)
    samples_cen_[i] = 0;
  auto &samples_reweight = samples_cen_;
  size_t n_samples_reweight = 0;
  for (size_t i = 0; i < n_samples; i++)
    if (Mah_dists2[i] < dist2_max)
    { for (size_t j = 0; j < n_dims; j++)
        samples_reweight[n_dims*n_samples_reweight+j] = samples[n_dims*i+j];
      n_samples_reweight++;
    }
    
// The mean and covariance (with the reweighting consistency factor applied) are
// the final MCD mean and covariance estimates. Careful, it is the norm to use a
// maximum likelihood sample covariance here; this just amounts to a norm for
// the reweighting factor.
  sample_mean_2(n_dims, n_samples_reweight, samples_reweight.data(), MCD_mean);
  samples_center(n_dims, n_samples_reweight, samples_reweight.data(), MCD_mean);
  sample_cov<true>(n_dims, n_samples_reweight, samples_reweight.data(), MCD_cov);
  for (size_t i = 0; i < pow(n_dims,2); i++)
    MCD_cov[i] *= c1;
}

} // stats
