////////////////////////////////////////////////////////////////////////
// Copyright (C) 2021 The Octave Project Developers
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////

#include <octave/oct.h>
#include <octave/lo-lapack-proto.h>

extern "C"
{
  F77_RET_T
  F77_FUNC (ztrexc, ZTREXC)
  (F77_CONST_CHAR_ARG_DECL, const F77_INT &, F77_DBLE_CMPLX *, const F77_INT &,
   F77_DBLE_CMPLX *, const F77_INT &, const F77_INT &, const F77_INT &,
   F77_INT &);

  F77_RET_T
  F77_FUNC (dtrexc, dTREXC)
  (F77_CONST_CHAR_ARG_DECL, const F77_INT &, F77_DBLE *, const F77_INT &,
   F77_DBLE *, const F77_INT &, const F77_INT &, const F77_INT &,
   F77_DBLE *,F77_INT &);

  F77_RET_T
  F77_FUNC (ctrexc, cTREXC)
  (F77_CONST_CHAR_ARG_DECL, const F77_INT &, F77_CMPLX *, const F77_INT &,
   F77_CMPLX *, const F77_INT &, const F77_INT &, const F77_INT &,
   F77_INT &);

  F77_RET_T
  F77_FUNC (strexc, sTREXC)
  (F77_CONST_CHAR_ARG_DECL, const F77_INT &, F77_REAL *, const F77_INT &,
   F77_REAL *, const F77_INT &, const F77_INT &, const F77_INT &,
   F77_REAL *,F77_INT &);
}

DEFUN_DLD (trexc, args, nargout, " -*- texinfo -*- \n\
             @deftypefn {Loadable Function} {[@var{UR}, @var{SR}] =} trexc (@var{U}, @var{S}, ifst, ilst) \n\
             Reorder Schur factorization @code{A = U * S * U'}, so that @var{S}(ifst,ifst) \n\
             is moved to @var{S}(ilst,ilst).  @var{UR} and @var{SR} form a further Schur factorization \n\
             of @var{A}.  trexc calls LAPACK routine trexc. \n\
             @end deftypefn ")
{
  // Essentially this code is a copy of libinterp/corefcn/ordschur.cc
  if (args.length () != 4)
    print_usage ();

  const dim_vector dimU = args (0).dims ();
  const dim_vector dimS = args (1).dims ();

  if (dimS (0) != dimS (1) || dimU (0) != dimU (1) || dimS (1) != dimU (1))
    error ("trexc: U and S must be square and of equal sizes");

  octave_value_list retval;

  const bool double_type
      = args (0).is_double_type () || args (1).is_double_type ();
  const bool complex_type = args (0).iscomplex () || args (1).iscomplex ();

#define PREPARE_ARGS(TYPE, TYPE_M, TYPE_COND)                                 \
  TYPE##Matrix U = args (0).x##TYPE_M##_value (                               \
      "trexc: U and S must be real or complex floating point matrices");      \
  TYPE##Matrix S = args (1).x##TYPE_M##_value (                               \
      "trexc: U and S must be real or complex floating point matrices");     \
   F77_INT info;

#define PREPARE_OUTPUT()                                                      \
  if (info != 0)                                                              \
    error ("trexc: trexc failed");                                            \
                                                                              \
  retval = ovl (U, S);

  F77_INT n = octave::to_f77_int (dimU (0));
  octave_value tmp = args (2);
  F77_INT ifst = octave::to_f77_int (tmp.idx_type_value (true));
  tmp = args (3);
  F77_INT ilst = octave::to_f77_int (tmp.idx_type_value (true));

  if (double_type)
    {
      if (complex_type)
        {
          PREPARE_ARGS (Complex, complex_matrix, double)

          F77_XFCN (ztrexc, ztrexc,
                    (F77_CONST_CHAR_ARG ("V"), n,
                     F77_DBLE_CMPLX_ARG (S.fortran_vec ()), n,
                     F77_DBLE_CMPLX_ARG (U.fortran_vec ()), n, ifst, ilst,
                     info));

          PREPARE_OUTPUT ()
        }
      else
        {
          PREPARE_ARGS (, matrix, double)
          Matrix work (dim_vector (n, 1));
          F77_XFCN (dtrexc, dtrexc,
                    (F77_CONST_CHAR_ARG ("V"), n,
                     S.fortran_vec (), n,
                     U.fortran_vec (), n, ifst, ilst,
                     work.fortran_vec (),info));

          PREPARE_OUTPUT ()
        }
    }
  else
    {
      if (complex_type)
        {
          PREPARE_ARGS (FloatComplex, float_complex_matrix, float)
          F77_XFCN (ctrexc, ctrexc,
                    (F77_CONST_CHAR_ARG ("V"), n,
                     F77_CMPLX_ARG (S.fortran_vec ()), n,
                     F77_CMPLX_ARG (U.fortran_vec ()), n, ifst, ilst,
                     info));

          PREPARE_OUTPUT ()
        }
      else
        {
          PREPARE_ARGS (Float, float_matrix, float) 
          FloatMatrix work (dim_vector (n, 1));
          F77_XFCN (strexc, strexc,
                    (F77_CONST_CHAR_ARG ("V"), n,
                     S.fortran_vec (), n,
                     U.fortran_vec (), n, ifst, ilst,
                     work.fortran_vec () ,info));

          PREPARE_OUTPUT ()
        }
    }

#undef PREPARE_ARGS
#undef PREPARE_OUTPUT

  return retval;
}

/*
%!test
%! randn ("state", 42);
%! A = randn(6);
%! [UOLD,TOLD] = schur(A,'complex');
%! old_diag = TOLD(3,3);
%! [UNEW1,TNEW1] = trexc(UOLD,TOLD,3,5);
%! new_diag = TNEW1(5,5);
%! assert (UNEW1 * TNEW1 * UNEW1', A, sqrt (eps));
%! assert (old_diag,new_diag,sqrt (eps));
*/
