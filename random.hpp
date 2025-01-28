#pragma once
#include <cmath>
#include <numbers>

// PCG XSL-RR generator, 64 bit output, 128 bit state. Adapted from the PCG
// library. Copyright M.E. O'Neill (2014), Apache License 2.0.
namespace pcg_xsl_rr_128_64
{

using namespace std::numbers;

// This is a CPU-optimized version, with the particular difference being the way
// normal variables are produced.
// There is the Marsaglia polar method, and the Box-Muller transform.
// The former is a Monte-Carlo algorithm, so not deterministic. The latter is
// deterministic but uses the sine and cosine, which is quite slow on both CPU
// and GPU. Both produce two samples at a time, so the random number generator
// has a state variable normal_has_spare, which could also be a static variable
// in the function. Practically it is optimized away on the CPU, so shouldn't
// take it seriously, and on the GPU it's done in parallel anyway so it's more
// elegant to produce two at a time.
struct Generator
{
  static constexpr auto mult = (static_cast<__uint128_t>(2549297995355413924)<<64)|4865540595714422341;
  __uint128_t state, inc;
  double normal_spare;
  bool normal_has_spare = false;

  Generator() {}

  Generator
  ( __uint128_t state_seed, __uint128_t inc_seed
  )
  { state = 0;
    inc = inc_seed<<1|1; // inc must be odd
    state = state*mult+inc;
    state += state_seed;
    state = state*mult+inc;
  }

  uint64_t next
  ()
  { state = state*mult+inc;
    uint64_t x = static_cast<uint64_t>(state>>64)^state;
    size_t rot = state>>122;
    return x>>rot|x<<(-rot&63);
  }

  double uniform
  ()
  { constexpr auto r_max_uint64_p1 = 1./(__uint128_t {1}<<64);
    return next()*r_max_uint64_p1;
  }

// Marsaglia polar method
  double normal
  ()
  { if (normal_has_spare)
    { normal_has_spare = false;
      return normal_spare;
    }
    double u1, u2, s;
    for (;;)
    { u1 = uniform()*2-1;
      u2 = uniform()*2-1;
      s = pow(u1,2)+pow(u2,2);
      if (s != 0 && s < 1)
// s != 0 is an important detail.
        break;
    }
    s = sqrt(-2*log(s)/s);
    normal_spare = u1*s;
    normal_has_spare = true;
    return u2*s;
  }

/*
// Box-Muller
  double normal
  ()
  { if (normal_has_spare)
    { normal_has_spare = false;
      return normal_spare;
    }
    double u1 = uniform(),
           u2 = uniform();
    double r = sqrt(-2*log(1-u1));
// u1 is [0,1), need (0,1], common mistake to use u1 instead of 1-u1 here.
    normal_spare = r*cos(2*pi*u2);
    return r*sin(2*pi*u1);
  }
*/

// Fisher-Yates. (Fails for n=0 but don't be an idiot is a prerequisite.)
  template <typename T>
  void permute
  ( size_t n, T *x
  )
  { for (size_t i = n-1; i > 0; i--)
      std::swap(x[i], x[next()%(i+1)]);
  }
};

} // pcg_xsl_rr_128_64
