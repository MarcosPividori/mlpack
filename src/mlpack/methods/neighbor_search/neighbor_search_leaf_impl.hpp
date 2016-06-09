#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_LEAF_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_LEAF_IMPL_HPP

// In case it hasn't been included yet.
#include "neighbor_search_leaf.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearchLeaf<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
NeighborSearchLeaf(const bool naive,
                   const bool singleMode,
                   const size_t leafSize) :
    leafSize(leafSize),
    ns(naive,singleMode)
{
    /* Nothing to do. */
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearchLeaf<SortPolicy, MetricType, MatType, TreeType,
TraversalType>::Train(MatType&& referenceSet)
{
  if (!ns.Naive())
  {
    std::vector<size_t> oldFromNewRef;
    Tree* tree = new Tree(std::move(referenceSet), oldFromNewRef, leafSize);
    ns.Train(tree);
    // Give the model ownership of the tree and the mappings.
    ns.treeOwner = true;
    ns.oldFromNewReferences = std::move(oldFromNewRef);
  }
  else
    ns.Train(std::move(referenceSet));
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearchLeaf<SortPolicy, MetricType, MatType, TreeType,
TraversalType>::Search(const MatType& querySet,
                       const size_t k,
                       arma::Mat<size_t>& neighbors,
                       arma::mat& distances)
{
  if (!ns.Naive() && !ns.SingleMode())
  {
    std::vector<size_t> oldFromNewQueries;
    Tree queryTree(std::move(querySet), oldFromNewQueries, leafSize);

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    ns.Search(&queryTree, k, neighborsOut, distancesOut);

    // Unmap the query points.
    distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
    neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
    for (size_t i = 0; i < neighborsOut.n_cols; ++i)
    {
      neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
      distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
    }
  }
  else
    ns.Search(querySet, k, neighbors, distances);
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearchLeaf<SortPolicy, MetricType, MatType, TreeType,
TraversalType>::Search(const size_t k,
                       arma::Mat<size_t>& neighbors,
                       arma::mat& distances)
{
  ns.Search(k,neighbors,distances);
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
template<typename Archive>
void NeighborSearchLeaf<SortPolicy, MetricType, MatType, TreeType,
    TraversalType>::Serialize(Archive& ar,
                              const unsigned int /* version */)
{
  ar & data::CreateNVP(leafSize, "leafSize");
  ar & data::CreateNVP(ns, "ns");
}

} // namespace neighbor
} // namespace mlpack

#endif
