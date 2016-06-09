#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_LEAF_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_LEAF_HPP

#include "neighbor_search_gen.hpp"
#include "neighbor_search.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy = NearestNeighborSort,
         typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::KDTree,
         template<typename RuleType> class TraversalType =
             TreeType<MetricType,
                      NeighborSearchStat<SortPolicy>,
                      MatType>::template DualTreeTraverser>
class NeighborSearchLeaf : public NeighborSearchGen<MatType>
{
 private:
  using Tree = TreeType<MetricType, NeighborSearchStat<SortPolicy>, MatType>;

  size_t leafSize;

  NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType> ns;

 public:
  NeighborSearchLeaf(const bool naive = false,
                     const bool singleMode = false,
                     const size_t leafSize = 20);

  ~NeighborSearchLeaf() {};

  void Train(MatType&& referenceSet);

  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  const MatType& ReferenceSet() const { return ns.ReferenceSet(); };

  bool Naive() const { return ns.Naive(); };
  bool& Naive() { return ns.Naive(); };

  bool SingleMode() const { return ns.SingleMode(); };
  bool& SingleMode() { return ns.SingleMode(); };

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

}; // class NeighborSearchLeaf

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "neighbor_search_leaf_impl.hpp"

#endif
