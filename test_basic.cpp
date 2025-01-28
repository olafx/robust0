/*
Tests sampling and the basic estimators of the mean and covariance, as well as
the Mahalanobis distance.
*/

#include "stats.hpp"
#include "sample.hpp"

int main()
{
  constexpr size_t n_dims = 3, n_samples = 1000;
  std::vector<double> mean {1, 2, 3}, cov {.3, .0, .1, .0, .2, .0, .0, .0, .1} /* lower triangular only */;
  pcg_xsl_rr_128_64::Generator gen {42, 69};
  std::vector<double> samples(n_dims*n_samples),
                      samples_cen(n_dims*n_samples),
                      sample_mean_1(n_dims, 0),
                      sample_mean_2(n_dims),
                      sample_cov(n_dims*n_dims),
                      Mah_dist2(n_samples, 0);
  sample::multivariate(gen, n_dims, mean.data(), cov.data(), n_samples, samples.data());
  stats::sample_mean_1(n_dims, n_samples, samples.data(), sample_mean_1.data());
  stats::sample_mean_2(n_dims, n_samples, samples.data(), sample_mean_2.data());
  stats::samples_cen(n_dims, n_samples, samples.data(), sample_mean_2.data(), samples_cen.data());
  stats::sample_cov(n_dims, n_samples, samples_cen.data(), sample_cov.data());
  stats::Mah_dist2(n_dims, n_samples, samples_cen.data(), sample_mean_2.data(), sample_cov.data(), Mah_dist2.data());
  std::println("mean\nactual   {::+.2e}\nestimate {::+.2e} (naive)\nestimate {::+.2e} (divide-and-conquer)", mean, sample_mean_1, sample_mean_2);
  std::println("cov\nactual   {::+.2e}\nestimate {::+.6e}", cov, sample_cov);
  std::println("1st sq Mah dist\n{:.2e}", Mah_dist2[0]);
  for (size_t i = 0; i < n_samples; i++)
    assert(Mah_dist2[i] >= 0);
  std::println("{::.2e}", Mah_dist2);
}
