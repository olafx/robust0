#include "stats.hpp"

int main()
{
  constexpr size_t n = 1e4;
  std::vector<double> x(n);
  auto reset = [&](size_t m)
  { for (size_t i = 0; i < m; i++)
      x[i] = i+1;
  };
  for (size_t m = 1; m < n; m++)
  { reset(m);
    if (m*(m+1)/2 != stats::sum_3(m, x.data()))
      std::println("method 3 failed {}", m);
    if (m*(m+1)/2 != stats::sum_1(m, x.data()))
      std::println("method 1 failed {}", m);
    reset(m);
    if (m*(m+1)/2 != stats::sum_2(m, x.data()))
      std::println("method 2 failed {}", m);    
  }
}
