#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_GEN_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_GEN_HPP

#include <mlpack/core.hpp>
#include <vector>
#include <string>

namespace mlpack {
namespace neighbor {

template<typename MatType = arma::mat>
class NeighborSearchGen
{
 public:
  virtual ~NeighborSearchGen() {};

  virtual void Train(MatType&& referenceSet) = 0;

  virtual void Search(const MatType& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances) = 0;

  virtual void Search(const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances) = 0;

  virtual const MatType& ReferenceSet() const = 0;

  virtual bool Naive() const = 0;
  virtual bool& Naive() = 0;

  virtual bool SingleMode() const = 0;
  virtual bool& SingleMode() = 0;

}; // class NeighborSearchGen

} // namespace neighbor
} // namespace mlpack

#endif
