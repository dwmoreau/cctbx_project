#include <boost/scoped_ptr.hpp>
#include <scitbx/error.h>
#include <scitbx/fftpack/real_to_complex.h>

#include <rstbx/dps_core/directional_fft.h>

rstbx::Directional_FFT::Directional_FFT (
  const Direction& angle, const veclist_t& xyzdata,
  const double& granularity, const double& amax,
  const sztype& F0_specific_cutoff, fft_workspace* workspace):
    xy(xyzdata.const_ref()),
    has_power_spectrum(false),
    has_kval(false),
    F0_specific_cutoff(F0_specific_cutoff){

      // A caller that supplies no workspace gets one for this transform alone,
      // so that both routes run the same code.
      boost::scoped_ptr<fft_workspace> owned;
      if (workspace == 0) {
        owned.reset(new fft_workspace());
        workspace = owned.get();
      }

      const sztype n_spots = xy.size();
      SCITBX_ASSERT(n_spots > 0);
      double* ps = workspace->space(workspace->projections, n_spots);

      for (sztype k=0; k < n_spots; ++k) {
        // unroll the dot product; 1% increase in efficiency
        ps[k] = xy[k][0]*angle.dvec[0]+xy[k][1]*angle.dvec[1]+xy[k][2]*angle.dvec[2];
      }

             pmin = *(std::min_element(ps, ps + n_spots));
      double pmax = *(std::max_element(ps, ps + n_spots));

      m = 1 + (sztype)((pmax - pmin) * granularity * amax );
      m = primecheck(m);
      m = std::max<int>(2,m); //Guard against extremely rare divide by zero with m==1

      delta_p = (pmax - pmin) / (m-1);

      fft_workspace::plan_t& FFT = workspace->plan(m);

      flexdouble frequency_distribution( scitbx::af::flex_grid<>(FFT.m_real()));
      double* f_d_ptr = frequency_distribution.begin();

      for (sztype k=0; k < n_spots; ++k) {
        f_d_ptr[  (int)(((ps[k]-pmin)/delta_p) + 0.5)  ]+=1.0;
      }

      FFT.forward(frequency_distribution.begin(),
                  workspace->space(workspace->scratch, m));
      fft_n_complex = FFT.n_complex();
      fft_result = flexcomplex(frequency_distribution.handle(),
                         scitbx::af::flex_grid<>((fft_n_complex)) );
}

rstbx::flexdouble
rstbx::Directional_FFT::power_spectrum() {
      if (!has_power_spectrum) {
        pspectrum = flexdouble(fft_n_complex, af::init_functor_null<double>() );
        double* psptr = pspectrum.begin();
        std::complex<double>* resptr = fft_result.begin();
        for (sztype k=0; k<fft_n_complex; ++k) {
          psptr[k]=std::abs(resptr[k]);
        }
        has_power_spectrum = true;
      }
      return pspectrum;
}

rstbx::sztype
rstbx::Directional_FFT::kmax() {
      if (!has_kval) {
        kval();
      }
      return p_kmax;
}

// Relative width of the band below the running maximum within which the
// squared magnitude cannot be trusted to order two channels. A gap wider than
// this in the squares is wider than the rounding of std::abs by a factor of
// several hundred.
static const double kval_tie_margin = 1.0e-13;

double
rstbx::Directional_FFT::kval() {
      if (!has_kval) {
        p_kmax = 0;
        p_kval = 0.0;
        double p_square = 0.0;
        std::complex<double>* resptr = fft_result.begin();
        for (sztype k = F0_specific_cutoff; k<fft_n_complex; ++k){
          double re = resptr[k].real();
          double im = resptr[k].imag();
          double square = re*re + im*im;
          // The squared magnitude orders the channels everywhere except within
          // a few ulps of the running maximum, where it can disagree with
          // std::abs. Those channels, and only those, are compared on std::abs.
          if ( square >= p_square * (1.0 - kval_tie_margin) ) {
            double value = std::abs(resptr[k]);
            if ( value > p_kval ) { p_kval=value; p_square=square; p_kmax=k;}
          }
        }
        has_kval = true;
      }
      return p_kval;
}

double
rstbx::Directional_FFT::kval0() {
      return std::abs(fft_result.begin()[0]);
}

double
rstbx::Directional_FFT::kval2() {
      if (!has_kval) { kval(); }
      if (p_kmax * 2 < fft_n_complex) {
        return std::abs(fft_result.begin()[p_kmax * 2]);
      }
      return 0.0;
}

double
rstbx::Directional_FFT::kval3() {
      if (!has_kval) { kval(); }
      if (p_kmax * 3 < fft_n_complex) {
        return std::abs(fft_result.begin()[p_kmax * 3]);
      }
      return 0.0;
}

rstbx::sztype
rstbx::primecheck(const sztype& input_m){
  sztype residual(input_m);
  sztype return_m(input_m);
  while (residual%2==0){ residual/=2; }
  while (residual%3==0){ residual/=3; }
  while (residual%5==0){ residual/=5; }
  if (residual>1) { return_m = primecheck(return_m+1); }
  return return_m;
}
