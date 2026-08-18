#ifndef RSTBX_INDEXING_API_SAMPLING_H
#define RSTBX_INDEXING_API_SAMPLING_H

#include <cmath>
#include <scitbx/constants.h>
#include <scitbx/error.h>
#include <scitbx/array_family/shared.h>
#include <scitbx/misc/positive_getitem_index.h>
#include <annlib_adaptbx/ann_adaptor.h>
#include <rstbx/dps_core/direction.h>

namespace rstbx { namespace indexing_api {

  /* Select the target grid directions that are near neighbors of the
     winning directions of a coarser first round of testing, so that the
     second round can restrict its FFTs to those directions.

     The returned directions keep the order they have in angles.
   */
  inline
  scitbx::af::shared<Direction>
  restrict_target_angles_to_quick_winners(
    scitbx::af::shared<Direction> const& angles,
    scitbx::af::shared<Direction> const& quick_sampling,
    double const& second_round_sampling,
    double const& quick_grid)
  {
    scitbx::af::shared<double> target;
    target.reserve(3 * angles.size());
    for (std::size_t iz = 0; iz < angles.size(); ++iz) {
      Direction::point const& v = angles[iz].dvec;
      target.push_back(v[0]);
      target.push_back(v[1]);
      target.push_back(v[2]);
    }
    int kn = static_cast<int>(2. * second_round_sampling);

    //construct k-d tree for the reference set
    annlib_adaptbx::AnnAdaptor A(target, 3, kn);

    scitbx::af::shared<double> query;
    query.reserve(3 * quick_sampling.size());
    for (std::size_t j = 0; j < quick_sampling.size(); ++j) {
      Direction::point const& v = quick_sampling[j].dvec;
      query.push_back(v[0]);
      query.push_back(v[1]);
      query.push_back(v[2]);
      if (std::fabs(scitbx::constants::pi_2 - quick_sampling[j].psi)
            < 0.0001) { //take care of equatorial boundary
        query.push_back(-v[0]);
        query.push_back(-v[1]);
        query.push_back(-v[2]);
      }
    }

    A.query(query); //find nearest neighbors of query points
    SCITBX_ASSERT(A.nn.size() == A.distances.size());

    scitbx::af::shared<bool> accept_flag(angles.size(), false);
    for (std::size_t idx = 0; idx < A.distances.size(); ++idx) {
      // use small angle approximation to test if target is within
      // desired radius
      if (std::sqrt(A.distances[idx]) < quick_grid) {
        accept_flag[scitbx::positive_getitem_index(
          A.nn[idx], accept_flag.size())] = true;
      }
    }

    //go through all of the original target angles
    scitbx::af::shared<Direction> restricted;
    for (std::size_t iz = 0; iz < angles.size(); ++iz) {
      if (accept_flag[iz]) {
        restricted.push_back(angles[iz]);
      }
    }
    return restricted;
  }

}} //namespace rstbx::indexing_api

#endif //RSTBX_INDEXING_API_SAMPLING_H
