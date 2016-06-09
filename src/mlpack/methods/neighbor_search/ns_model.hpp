/**
 * @file ns_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include "neighbor_search_gen.hpp"
#include "neighbor_search.hpp"
#include "neighbor_search_leaf.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy>
struct NSModelName
{
  static const std::string Name() { return "neighbor_search_model"; }
};

template<>
const std::string NSModelName<NearestNeighborSort>::Name()
{
  return "nearest_neighbor_search_model";
}

template<>
const std::string NSModelName<FurthestNeighborSort>::Name()
{
  return "furthest_neighbor_search_model";
}

template<typename SortPolicy>
class NSModel
{
 public:
  enum TreeTypes
  {
    KD_TREE,
    COVER_TREE,
    R_TREE,
    R_STAR_TREE,
    BALL_TREE,
    X_TREE
  };

 private:
  TreeTypes treeType;

  bool randomBasis;
  arma::mat q;

  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using NSType = NeighborSearch<SortPolicy,
                                metric::EuclideanDistance,
                                arma::mat,
                                TreeType,
                                TreeType<metric::EuclideanDistance,
                                    NeighborSearchStat<SortPolicy>,
                                    arma::mat>::template DualTreeTraverser>;

  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using NSLeaf = NeighborSearchLeaf<SortPolicy,
                                    metric::EuclideanDistance,
                                    arma::mat,
                                    TreeType,
                                    TreeType<metric::EuclideanDistance,
                                        NeighborSearchStat<SortPolicy>,
                                        arma::mat>::template DualTreeTraverser>;

  NeighborSearchGen<arma::mat>* nSearch;

 public:
  NSModel(TreeTypes treeTyp = KD_TREE, bool randomBasis = false);

  ~NSModel();

  void BuildModel(arma::mat&& referenceSet,
                  const size_t leafSize = 20,
                  const bool naive = false,
                  const bool singleMode = false);

  void Search(arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  const arma::mat& Dataset() const;

  bool Naive() const;
  bool& Naive();

  bool SingleMode() const;
  bool& SingleMode();

  std::string TreeName() const;

  //! Serialize the neighbor search model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);
};

using KNNModel = NSModel<NearestNeighborSort>;
using KFNModel = NSModel<FurthestNeighborSort>;

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "ns_model_impl.hpp"

#endif
