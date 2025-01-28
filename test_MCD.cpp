#include "stats.hpp"
#include "sample.hpp"
#include <gsl/gsl_cdf.h>

int main()
{
  constexpr size_t n_dims = 3, n_samples = 10000, n_outliers = 100, h = n_samples*0.8;
  std::vector<double> mean          {1, 2, 3},
                      mean_outliers {7, 4, -3} /* somewhere very different */,
                      cov          { .3, .0, .1, .0, .2, .0, .0, .0, .1} /* lower triangular specified only */,
                      cov_outliers { .2, .0, .1, .0, .2, .0, .0, .0, .2} /* a bit different */;
  pcg_xsl_rr_128_64::Generator gen {42, 69};
  std::vector<double> samples(n_dims*n_samples),
                      samples_cen(n_dims*n_samples),
                      sample_mean_1(n_dims),
                      sample_mean_2(n_dims),
                      sample_cov_1(pow(n_dims,2)),
                      sample_cov_2(pow(n_dims,2));
  sample::multivariate(gen, n_dims, mean.data(), cov.data(), n_samples-n_outliers, samples.data());
  sample::multivariate(gen, n_dims, mean_outliers.data(), cov_outliers.data(), n_outliers, samples.data()+(n_samples-n_outliers)*n_dims);
// All algorithms are independent of sample ordering, but just to be sure, mix
// it all up.
  stats::permute_samples_2(gen, n_dims, n_samples, samples.data());
  stats::sample_mean_2(n_dims, n_samples, samples.data(), sample_mean_1.data());
  stats::samples_cen(n_dims, n_samples, samples.data(), sample_mean_1.data(), samples_cen.data());
  stats::sample_cov(n_dims, n_samples, samples_cen.data(), sample_cov_1.data());
// Test the partial sorting algorithm.
  std::vector<double> x {1., 1., 1., .1, 3., 3., 4., .1, .3, .2, 5., 6.};
  std::vector<size_t> x_ids(3);
  stats::partial_sort_smallest(x_ids.size(), x.size(), x.data(), x_ids.data());
  std::println("{::}", x_ids);
// Test quantiles. (Not that this must be particularly accurate in any case for
// MCD.)
// >>> scipy.stats.chi2.ppf(1-0.025, 3)
// np.float64(9.348403604496145)
  std::println("{:.16e}", gsl_cdf_chisq_Pinv(1-0.025, 3));
// And finally, the MCD.
  stats::MCD(gen, n_dims, n_samples, samples.data(), h, sample_mean_2.data(), sample_cov_2.data());
  std::println("mean\nactual   {::+.2e}\nestimate {::+.2e} (non-robust)\nestimate {::+.2e} (robust)", mean, sample_mean_1, sample_mean_2);
  std::println("cov\nactual   {::+.2e}\nestimate {::+.2e} (non-robust)\nestimate {::+.2e} (robust)", cov, sample_cov_1, sample_cov_2);
}
