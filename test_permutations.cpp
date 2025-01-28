#include <print>
#include <vector>
#include <string>
#include "random.hpp"
#include "stats.hpp"

int main()
{
  constexpr size_t n_seeds = 10, n_dims = 3, n_samples = 10, n_samples_out = 4;
  std::vector<std::string> x(n_samples);
  for (size_t i = 0; i < n_seeds; i++)
  { std::println("{}/{}", i+1, n_seeds);
// Weird seeds, keep them distinct. Also warm up the RNGs.
    pcg_xsl_rr_128_64::Generator gen {42+i*i+2*i, 69+3*i};
    for (size_t j = 0; j < 10; j++)
      gen.next();
    for (size_t j = 0; j < n_samples; j++)
      x[j] = std::to_string(j+1);
    std::println("{}", x);
    gen.permute(n_samples, x.data());
    std::println("{}", x);
  }
// Also try the slightly different algorithms acting on samples.
  std::vector<double> samples(n_samples*n_dims),
                      samples_out(n_samples_out*n_dims);
  auto init_samples = [&]()
  { for (size_t i = 0; i < n_samples; i++)
      for (size_t j = 0; j < n_dims; j++)
        samples[n_dims*i+j] = i;
  };
  init_samples();
  std::println("original {::}", samples);
  pcg_xsl_rr_128_64::Generator gen {42, 69};
  stats::permute_samples_1(gen, n_dims, n_samples, samples.data());
  std::println("permuted {::} (in-place)", samples);
  init_samples();
  gen = pcg_xsl_rr_128_64::Generator {42, 69};
  stats::permute_samples_2(gen, n_dims, n_samples, samples.data());
  std::println("permuted {::} (out-of-place)", samples);
// And the subset permutations.
  init_samples();
  gen = pcg_xsl_rr_128_64::Generator {42, 69};
  for (size_t i = 0; i < 4; i++)
  { stats::permute_subset_samples(gen, n_dims, n_samples, samples.data(), n_samples_out, samples_out.data());
    std::println("permuted {::} (out-of-place subset of {})", samples_out, n_samples_out);
  }
}
