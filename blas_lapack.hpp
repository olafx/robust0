#pragma once

namespace blas_lapack
{

extern "C"
{

extern void dsyrk_
( const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, const double *a, const int *lda,
  const double *beta, double *c, const int *ldc
);
extern void dlacpy_
( const char *uplo, const int *m, const int *n, const double *a, const int *lda, double *b, const int *ldb
);
extern void dpotrf_
( const char *uplo, const int *n, double *a, const int *lda, int *info
);
extern void dtrmv_
( const char *uplo, const char *trans, const char *diag, const int *n, const double *a, const int *lda, double *x,
  const int *incx
);
extern void dgemm_
( const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha,
  const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, int *ldc
);
extern void dgemv_
( const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x,
  const int *incx, const double *beta, double *y, const int *incy
);
extern double ddot_
( const int *n, const double *dx, const int *incx, double *dy, const int *incy
);
extern void dsymv_
( const char *uplo, const int *n, const double *alpha, const double *a, const int *lda, const double *x,
  const int *incx, const double *beta, double *y, const int *incy
);
extern void dsytrf_
( const char *uplo, const int *n, double *a, const int *lda, int *ipiv, double *work, const int *lwork, int *info
);
extern void dsytri_
( const char *uplo, const int *n, double *a, const int *lda, const int *ipiv, double *work, int *info
);

}

} // blas_lapack
