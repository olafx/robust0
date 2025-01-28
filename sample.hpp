#pragma once
#include <cmath>
#include <cassert>
#include <vector>
#include "random.hpp"
#include "blas_lapack.hpp"

namespace sample
{

void multivariate
( pcg_xsl_rr_128_64::Generator &gen, size_t n_dims, const double *mean, const double *cov, size_t n_samples,
  double *samples
)
{ int n_dims_ = n_dims, lp_info;
  std::vector<double> Ch_L(n_dims*n_dims, 0);
  blas_lapack::dlacpy_("L", &n_dims_, &n_dims_, cov, &n_dims_, Ch_L.data(), &n_dims_);
  blas_lapack::dpotrf_("L", &n_dims_, Ch_L.data(), &n_dims_, &lp_info);
  assert(!lp_info);
  for (size_t i = 0; i < n_dims*n_samples; i++)
    samples[i] = gen.normal();
  constexpr int inc = 1;
  for (size_t i = 0; i < n_samples; i++)
  { blas_lapack::dtrmv_("L", "N", "N", &n_dims_, Ch_L.data(), &n_dims_, samples+n_dims*i, &inc);
    for (size_t j = 0; j < n_dims; j++)
      samples[i*n_dims+j] += mean[j];
  }
}

} // sample_multivariate
