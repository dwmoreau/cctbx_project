from __future__ import absolute_import, division, print_function
from six.moves import range
import random
import struct
from libtbx.test_utils import approx_equal
from rstbx.dps_core import Direction, Directional_FFT, dps_core
from scitbx import fftpack
from scitbx.array_family import flex

# dps_core::fft_factory builds every direction with the file-scope F0_cutoff,
# so the directly constructed transforms below must be given the same value.
F0_CUTOFF = 3

def primecheck(input_m):
  """Statement of rstbx::primecheck in Python.

  Objective: know the transform length the reference pipeline must use.
  Means: raise the length until it is 5-smooth. Expected result: the same
  regular number the compiled code selects."""
  m = input_m
  while True:
    residual = m
    for factor in (2,3,5):
      while residual % factor == 0:
        residual //= factor
    if residual <= 1:
      return m
    m += 1

def python_reference(dvec,xyzdata,granularity,amax,f0_cutoff=F0_CUTOFF):
  """Statement of the Directional_FFT pipeline in pure Python.

  Objective: provide a comparison standard that shares no code with the
  implementation. Means: project the spots onto the direction, histogram the
  projections over a 5-smooth number of channels, transform, and take the
  largest magnitude above the cutoff. Expected result: a dictionary of every
  quantity the class publishes."""
  ps = [v[0]*dvec[0] + v[1]*dvec[1] + v[2]*dvec[2] for v in xyzdata]
  pmin = min(ps)
  pmax = max(ps)
  m = max(2,primecheck(1 + int((pmax-pmin)*granularity*amax)))
  delta_p = (pmax-pmin)/(m-1)
  plan = fftpack.real_to_complex(m)
  histogram = flex.double(flex.grid(plan.m_real()))
  for p in ps:
    histogram[int(((p-pmin)/delta_p) + 0.5)] += 1.0
  plan.forward(histogram)
  bins = [complex(histogram[2*k],histogram[2*k+1])
          for k in range(plan.n_complex())]
  spectrum = [abs(z) for z in bins]
  kval = 0.0
  kmax = 0
  for k in range(f0_cutoff,len(spectrum)):
    if spectrum[k] > kval:
      kval = spectrum[k]
      kmax = k
  return dict(m=m, pmin=pmin, delta_p=delta_p, kval=kval, kmax=kmax,
              spectrum=spectrum, bins=bins)

def same_double(a,b):
  """Compare the IEEE-754 representation of two doubles.

  Byte comparison rather than a tolerance, because the property under test is
  an identical representation rather than an approximate value."""
  return struct.pack("<d",a) == struct.pack("<d",b)

def same_complex_array(a,b):
  """Compare two complex arrays by the representation of every component."""
  if a.size() != b.size():
    return False
  for i in range(a.size()):
    if not same_double(a[i].real,b[i].real): return False
    if not same_double(a[i].imag,b[i].imag): return False
  return True

def same_double_array(a,b):
  """Compare two double arrays by the representation of every element."""
  if a.size() != b.size():
    return False
  for i in range(a.size()):
    if not same_double(a[i],b[i]): return False
  return True

def fresh_transform(dvec,xyzdata,granularity,amax):
  """Evaluate one direction through a transform that shares nothing.

  The Python constructor of Directional_FFT takes no workspace, so it builds
  its own plan and buffers and discards them; this is the comparison standard
  for a direction taken from an engine whose workspace persists."""
  return Directional_FFT(angle=Direction(dvec), xyzdata=xyzdata,
                         granularity=granularity, amax=amax,
                         F0_cutoff=F0_CUTOFF)

def engine(xyzdata,amax):
  ai = dps_core()
  ai.setMaxcell(amax)
  ai.setXyzData(xyzdata)
  return ai

def assert_agrees(direction,transform,context):
  """Require a Direction and a directly built transform to agree exactly.

  The power spectrum is compared too: it is the whole transform output seen
  through a magnitude, so it pins far more than the five scalars do."""
  for name,got,expected in (
      ("kval",  direction.kval,    transform.kval()),
      ("kval0", direction.kval0,   transform.kval0()),
      ("kval2", direction.kval2,   transform.kval2()),
      ("kval3", direction.kval3,   transform.kval3()),
      ("delta_p", direction.delta_p, transform.delta_p),
      ("real",  direction.real,
                transform.kmax()/(transform.m*transform.delta_p))):
    assert same_double(got,expected), \
      "%s: %s %r != %r"%(name,context,got,expected)
  assert direction.kmax == transform.kmax(), context
  assert direction.m == transform.m, context
  assert same_double_array(direction.getff(),transform.power_spectrum()), context

def random_spots(count,seed):
  """Reciprocal-space vectors with no lattice, for a spread of transform sizes."""
  rng = random.Random(seed)
  return flex.vec3_double([(rng.uniform(-0.3,0.3),rng.uniform(-0.3,0.3),
                            rng.uniform(-0.3,0.3)) for _ in range(count)])

def random_directions(count,seed):
  rng = random.Random(seed)
  out = []
  while len(out) < count:
    v = (rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-1,1))
    length = (v[0]**2+v[1]**2+v[2]**2)**0.5
    if 0.2 < length <= 1.0:
      out.append((v[0]/length,v[1]/length,v[2]/length))
  return out

def lattice_spots(a,b,c,extent=4):
  """Reciprocal-lattice vectors of an orthorhombic cell."""
  return flex.vec3_double([(h/a,k/b,l/c)
                           for h in range(-extent,extent+1)
                           for k in range(-extent,extent+1)
                           for l in range(-extent,extent+1)
                           if (h,k,l) != (0,0,0)])

def exercise_workspace_reuse():
  """A persistent workspace must not change any published value.

  This is the property the workspace has to have: an engine that keeps its
  plans and scratch buffers across thousands of directions must publish, for
  every one of them, exactly what a transform built from scratch publishes."""
  xyzdata = random_spots(400,seed=17)
  amax = 320.0
  ai = engine(xyzdata,amax)
  directions = random_directions(300,seed=23)

  sizes = set()
  for i,dvec in enumerate(directions):
    got = ai.fft_result(Direction(dvec))
    expected = fresh_transform(dvec,xyzdata,ai.granularity,amax)
    assert_agrees(got,expected,"direction %d"%i)
    sizes.add(got.m)
  assert len(sizes) > 1, "the test never exercised a second transform length"

  # the same engine, the same directions in the opposite order: a cache whose
  # contents depended on the order of arrival would show up here
  reversed_engine = engine(xyzdata,amax)
  forward = [ai.fft_result(Direction(d)).kval for d in directions]
  backward = [reversed_engine.fft_result(Direction(d)).kval
              for d in reversed(directions)]
  backward.reverse()
  for i in range(len(forward)):
    assert same_double(forward[i],backward[i]), "order dependence at %d"%i
  return len(sizes)

def exercise_python_reference():
  """Both compiled routes must agree with an independent statement of the
  algorithm, so that they cannot be wrong together."""
  xyzdata = random_spots(150,seed=11)
  amax = 250.0
  ai = engine(xyzdata,amax)
  for dvec in random_directions(12,seed=31):
    reference = python_reference(dvec,xyzdata,ai.granularity,amax)
    got = ai.fft_result(Direction(dvec))
    assert got.m == reference["m"]
    assert got.kmax == reference["kmax"]
    assert approx_equal(got.kval,reference["kval"],eps=1e-9)
    assert approx_equal(got.delta_p,reference["delta_p"],eps=1e-12)
    transform = fresh_transform(dvec,xyzdata,ai.granularity,amax)
    spectrum = transform.power_spectrum()
    assert spectrum.size() == len(reference["spectrum"])
    for k in range(spectrum.size()):
      assert approx_equal(spectrum[k],reference["spectrum"][k],eps=1e-9)
    # The compiled projection is a fused multiply-add of three terms and the
    # Python one is not, so the two agree to about an ulp rather than exactly.
    # It is far too small to move a spot between channels, which is why the
    # transform output below still matches bit for bit.
    assert approx_equal(transform.pmin,reference["pmin"],
                        eps=8*abs(reference["pmin"])*2.0**-52)
    assert same_complex_array(transform.fft_result,
                              flex.complex_double(reference["bins"]))

def exercise_kval_is_the_spectrum_maximum():
  """kval, kval0, kval2 and kval3 must stay the power spectrum's own values."""
  xyzdata = random_spots(200,seed=13)
  amax = 300.0
  ai = engine(xyzdata,amax)
  for dvec in random_directions(25,seed=37):
    ai.fft_result(Direction(dvec))
    transform = fresh_transform(dvec,xyzdata,ai.granularity,amax)
    spectrum = transform.power_spectrum()
    kmax = transform.kmax()
    assert kmax >= F0_CUTOFF or transform.kval() == 0.0
    best = 0.0
    for k in range(F0_CUTOFF,spectrum.size()):
      best = max(best,spectrum[k])
    assert same_double(transform.kval(),best)
    assert same_double(transform.kval0(),spectrum[0])
    assert same_double(transform.kval2(),
                       spectrum[kmax*2] if kmax*2 < spectrum.size() else 0.0)
    assert same_double(transform.kval3(),
                       spectrum[kmax*3] if kmax*3 < spectrum.size() else 0.0)

def exercise_degenerate_cases():
  """The smallest inputs the class accepts, on a workspace and without one."""
  # a spread so small that the channel count collapses onto the m>=2 clamp
  narrow = flex.vec3_double([(1.0,0.0,0.0),(1.0,0.0,1.0e-4)])
  ai = engine(narrow,320.0)
  got = ai.fft_result(Direction((0.0,0.0,1.0)))
  assert got.m == 2, got.m
  assert_agrees(got,fresh_transform((0.0,0.0,1.0),narrow,ai.granularity,320.0),
                "m==2 clamp")

  # a cutoff at or beyond the last channel leaves the maximum unset
  transform = Directional_FFT(angle=Direction((0.0,0.0,1.0)),xyzdata=narrow,
                              granularity=5.0,amax=320.0,F0_cutoff=64)
  assert transform.kval() == 0.0
  assert transform.kmax() == 0

  # two spots, one direction: a workspace holding exactly one plan
  pair = flex.vec3_double([(0.05,0.0,0.0),(-0.05,0.02,0.01)])
  single = engine(pair,120.0)
  only = single.fft_result(Direction((0.0,0.0,1.0)))
  assert_agrees(only,fresh_transform((0.0,0.0,1.0),pair,single.granularity,120.0),
                "single direction")

def exercise_lattice_recovery():
  """A known lattice must come back as a known cell length.

  Objective: pin what the class computes, not only that two routes agree.
  The tolerance is relative because the recovered length is quantised by the
  channel width, which is a few percent of the length at these grid sizes."""
  a,b,c = 40.0,55.0,70.0
  xyzdata = lattice_spots(a,b,c)
  ai = engine(xyzdata,100.0)
  for axis,expected in (((1.,0.,0.),a),((0.,1.,0.),b),((0.,0.,1.),c)):
    got = ai.fft_result(Direction(axis))
    assert abs(got.real-expected)/expected < 0.05, (axis,got.real,expected)

def exercise_comparators():
  """Negative control on the comparisons the rest of the file relies on."""
  assert same_double(0.1,0.1)
  assert not same_double(1.0,1.0 + 2.0**-52)
  assert not same_double(0.0,-0.0)
  one = flex.complex_double([complex(1.0,2.0),complex(3.0,4.0)])
  assert same_complex_array(one,one.deep_copy())
  changed = one.deep_copy()
  changed[1] = complex(3.0 + 2.0**-51,4.0)
  assert not same_complex_array(one,changed)
  assert not same_complex_array(one,flex.complex_double([complex(1.0,2.0)]))

def run():
  exercise_comparators()
  distinct = exercise_workspace_reuse()
  exercise_python_reference()
  exercise_kval_is_the_spectrum_maximum()
  exercise_degenerate_cases()
  exercise_lattice_recovery()
  print("OK (%d distinct transform lengths exercised)"%distinct)

if __name__ == "__main__":
  run()
