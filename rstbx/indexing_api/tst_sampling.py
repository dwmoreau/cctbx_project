from __future__ import absolute_import, division, print_function
from six.moves import range
import math
import struct
from libtbx.test_utils import approx_equal
from rstbx.array_family import flex
from rstbx.dps_core import SimpleSamplerTool
from rstbx.indexing_api import restrict_target_angles_to_quick_winners
from rstbx.indexing_api.sampling import HemisphereSampler
from annlib_ext import AnnAdaptor

def python_reference(angles,quick_sampling,second_round_sampling,quick_grid):
  """Statement of the restriction algorithm in pure Python.

  Objective: provide the comparison standard for the compiled
  implementation. Means: build a k-d tree over the target directions,
  query it with the quick sampling winners and their equatorial antipodes,
  and keep every target returned within quick_grid. Expected result: the
  accepted directions, in the order they occur in angles."""
  restricted = flex.Direction()
  target = flex.double()
  for iz in angles:
    v = iz.dvec
    target.append(v[0]);target.append(v[1]);target.append(v[2]);
  kn = int(2 * second_round_sampling)
  A = AnnAdaptor(data = target, dim = 3, k = kn)
  query = flex.double()
  for j in quick_sampling:
    v = j.dvec
    query.append(v[0]);query.append(v[1]); query.append(v[2]);
    if abs((math.pi/2.) - j.psi) < 0.0001:
      query.append(-v[0]);query.append(-v[1]); query.append(-v[2]);
  A.query(query)
  neighbors = flex.sqrt(A.distances)
  neighborid = A.nn
  accept_flag = flex.bool(len(angles))
  for idx in range(len(neighbors)):
    if neighbors[idx] < quick_grid:
      accept_flag[neighborid[idx]] = True
  for iz in range(len(angles)):
    if accept_flag[iz]:
      restricted.append(angles[iz])
  return restricted

def same_double(a,b):
  """Compare the IEEE-754 representation of two doubles.

  Byte comparison rather than a tolerance, because the property under test
  is an identical representation rather than an approximate value. It also
  makes two copies of the same NaN compare equal, which matters because
  kval0, kval2 and kval3 are unset on directions that have not been through
  a Fourier transform."""
  return struct.pack("<d",a) == struct.pack("<d",b)

# every member of Direction that is visible from Python, from the class
# wrapper in rstbx/ext.cpp
DOUBLE_ATTRIBUTES = ("psi","phi","kval","kval0","kval2","kval3",
                     "delta_p","real")
INTEGER_ATTRIBUTES = ("kmax","m")

def assert_identical(result,reference):
  """Require two direction lists to agree in every value visible from Python."""
  assert len(result) == len(reference), "%d != %d"%(len(result),len(reference))
  for i in range(len(reference)):
    r = result[i]
    e = reference[i]
    for name in DOUBLE_ATTRIBUTES:
      assert same_double(getattr(r,name),getattr(e,name)), \
        "%s of direction %d: %r != %r"%(name,i,getattr(r,name),getattr(e,name))
    for name in INTEGER_ATTRIBUTES:
      assert getattr(r,name) == getattr(e,name), \
        "%s of direction %d: %r != %r"%(name,i,getattr(r,name),getattr(e,name))
    for k in range(3):
      assert same_double(r.dvec[k],e.dvec[k]), \
        "dvec[%d] of direction %d: %r != %r"%(k,i,r.dvec[k],e.dvec[k])
      assert same_double(r.bvec()[k],e.bvec()[k]), \
        "bvec()[%d] of direction %d: %r != %r"%(k,i,r.bvec()[k],e.bvec()[k])
    rff = r.getff()
    eff = e.getff()
    assert len(rff) == len(eff)
    for k in range(len(eff)):
      assert same_double(rff[k],eff[k])

def compare(angles,quick_sampling,second_round_sampling,quick_grid):
  """Run both implementations on the same input and require identical output."""
  reference = python_reference(
    angles,quick_sampling,second_round_sampling,quick_grid)
  result = restrict_target_angles_to_quick_winners(
    angles = angles,
    quick_sampling = quick_sampling,
    second_round_sampling = second_round_sampling,
    quick_grid = quick_grid)
  assert_identical(result,reference)
  return result

def target_grid(sampling):
  """A hemisphere of directions at the given spacing in radians."""
  return SimpleSamplerTool(sampling).construct_hemisphere_grid(sampling)

def exercise_representative_case():
  """Compare on a finely sampled grid restricted by a coarse winner set."""
  angles = target_grid(0.05)
  coarse = target_grid(0.3)
  quick_sampling = flex.Direction(
    [coarse[i] for i in range(0,len(coarse),7)])
  quick_grid = 0.15
  second_round_sampling = math.pi*quick_grid*quick_grid/(0.05*0.05)
  result = compare(
    angles,quick_sampling,second_round_sampling,quick_grid)
  assert 0 < len(result) < len(angles)
  for i in range(len(result)):
    v = result[i].dvec
    assert approx_equal(math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]),1.0)

def exercise_production_geometry():
  """Compare on the grid parameters a 256 panel detector indexes with."""
  H = HemisphereSampler(
    max_grid = 0.029, characteristic_grid = 0.01565, quick_grid = 0.016)
  coarse = H.construct_hemisphere_grid(H.quick_grid)
  n = H.target_size_for_quick_grid_answer_vector
  quick_sampling = flex.Direction([coarse[i] for i in range(n)])
  angles = H.construct_hemisphere_grid(H.incr)
  result = compare(
    angles,quick_sampling,H.second_round_sampling,H.quick_grid)
  assert 0 < len(result) < len(angles)

def exercise_equatorial_antipodes():
  """Winners on the equator also contribute their antipodes to the query."""
  angles = target_grid(0.1)
  coarse = target_grid(0.4)
  quick_sampling = flex.Direction(
    [j for j in coarse if abs((math.pi/2.) - j.psi) < 0.0001])
  assert len(quick_sampling) > 0
  result = compare(angles,quick_sampling,math.pi*0.16/0.01,0.2)
  assert 0 < len(result) < len(angles)

def exercise_no_directions_accepted():
  """A zero acceptance radius retains nothing."""
  angles = target_grid(0.3)
  quick_sampling = target_grid(0.6)
  assert len(compare(angles,quick_sampling,10.,0.0)) == 0

def exercise_empty_quick_sampling():
  """An empty winner set produces an empty query and retains nothing."""
  angles = target_grid(0.3)
  assert len(compare(angles,flex.Direction(),10.,1.0)) == 0

def exercise_zero_neighbors_requested():
  """A request for zero neighbors returns an empty neighbor list."""
  angles = target_grid(0.3)
  quick_sampling = target_grid(0.6)
  assert len(compare(angles,quick_sampling,0.4,1.0)) == 0

def exercise_single_direction():
  """A target grid of one direction queried with that same direction.

  The search tree excludes exact self matches, so the one neighbor slot
  comes back unfilled and nothing is retained."""
  angles = flex.Direction([target_grid(0.3)[0]])
  assert len(compare(angles,angles,0.5,1.0)) == 0

def exercise_null_neighbor_index():
  """An unfilled neighbor slot reports index -1, which counts from the end.

  Three targets are queried with the first of them and three neighbors are
  requested. The self match is excluded, so the third slot is unfilled and
  reports -1 at an infinite distance. An acceptance radius above that
  distance makes the negative index reachable, and it selects the last
  target rather than the first."""
  full = target_grid(0.3)
  angles = flex.Direction([full[i] for i in range(3)])
  quick_sampling = flex.Direction([full[0]])
  assert len(compare(angles,quick_sampling,1.5,1.e300)) == 2

def exercise_empty_target_grid():
  """An empty target grid is rejected while the search tree is built."""
  messages = []
  calls = [
    lambda: python_reference(flex.Direction(),flex.Direction(),1.,1.),
    lambda: restrict_target_angles_to_quick_winners(
      angles = flex.Direction(),
      quick_sampling = flex.Direction(),
      second_round_sampling = 1.,
      quick_grid = 1.)]
  for call in calls:
    try:
      call()
    except RuntimeError as e:
      messages.append(str(e))
  assert len(messages) == 2, messages
  assert messages[0] == messages[1], messages
  assert messages[0].find("SCITBX_ASSERT(data.size() > 0) failure") >= 0

def exercise_method_contract():
  """The sampler method sets the restricted directions and adopts them.

  angles is a member of the C++ base class and assigning to it stores a
  copy, so the two agree in value rather than in identity."""
  H = HemisphereSampler(
    max_grid = 0.029, characteristic_grid = 0.05, quick_grid = 0.3)
  coarse = H.construct_hemisphere_grid(H.quick_grid)
  quick_sampling = flex.Direction(
    [coarse[i] for i in range(0,len(coarse),7)])
  H.angles = H.construct_hemisphere_grid(H.incr)
  reference = python_reference(
    H.angles,quick_sampling,H.second_round_sampling,H.quick_grid)
  assert H.restrict_target_angles_to_quick_winners(quick_sampling) is None
  assert_identical(H.restricted,reference)
  assert_identical(H.angles,reference)

def run():
  exercise_representative_case()
  exercise_production_geometry()
  exercise_equatorial_antipodes()
  exercise_no_directions_accepted()
  exercise_empty_quick_sampling()
  exercise_zero_neighbors_requested()
  exercise_single_direction()
  exercise_null_neighbor_index()
  exercise_empty_target_grid()
  exercise_method_contract()
  print("OK")

if (__name__ == "__main__"):
  run()
