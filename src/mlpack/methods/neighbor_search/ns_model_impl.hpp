/**
 * @file ns_model_impl.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ns_model.hpp"

namespace mlpack {
namespace neighbor {

/**
 * Initialize the NSModel with the given type and whether or not a random
 * basis should be used.
 */
template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    randomBasis(randomBasis),
    nSearch(NULL)
{
}

//! Clean memory, if necessary.
template<typename SortPolicy>
NSModel<SortPolicy>::~NSModel()
{
  if (nSearch)
    delete nSearch;
}

template<typename SortPolicy>
void NSModel<SortPolicy>::BuildModel(arma::mat&& referenceSet,
                                     const size_t leafSize,
                                     const bool naive,
                                     const bool singleMode)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    while (true)
    {
      // [Q, R] = qr(randn(d, d));
      // Q = Q * diag(sign(diag(R)));
      arma::mat r;
      if (arma::qr(q, r, arma::randn<arma::mat>(referenceSet.n_rows,
              referenceSet.n_rows)))
      {
        arma::vec rDiag(r.n_rows);
        for (size_t i = 0; i < rDiag.n_elem; ++i)
        {
          if (r(i, i) < 0)
            rDiag(i) = -1;
          else if (r(i, i) > 0)
            rDiag(i) = 1;
          else
            rDiag(i) = 0;
        }

        q *= arma::diagmat(rDiag);

        // Check if the determinant is positive.
        if (arma::det(q) >= 0)
          break;
      }
    }
    referenceSet = q * referenceSet;
  }

  if (nSearch)
    delete nSearch;

  switch (treeType)
  {
    case KD_TREE:
      nSearch = new NSLeaf<tree::KDTree>(naive, singleMode, leafSize);
      break;
    case COVER_TREE:
      nSearch = new NSType<tree::StandardCoverTree>(naive, singleMode);
      break;
    case R_TREE:
      nSearch = new NSType<tree::RTree>(naive, singleMode);
      break;
    case R_STAR_TREE:
      nSearch = new NSType<tree::RStarTree>(naive, singleMode);
      break;
    case BALL_TREE:
      nSearch = new NSLeaf<tree::BallTree>(naive, singleMode, leafSize);
      break;
    case X_TREE:
      nSearch = new NSType<tree::XTree>(naive, singleMode);
      break;
  }
  if (!nSearch)
    throw std::runtime_error("Couldn't create NeighborSearch object");

  nSearch->Train(std::move(referenceSet));
}

template<typename SortPolicy>
void NSModel<SortPolicy>::Search(arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
    querySet = q * querySet;
  if (nSearch)
    return nSearch->Search(querySet, k, neighbors, distances);
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
void NSModel<SortPolicy>::Search(const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  if (nSearch)
    return nSearch->Search(k, neighbors, distances);
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
const arma::mat& NSModel<SortPolicy>::Dataset() const
{
  if (nSearch)
    return nSearch->ReferenceSet();
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool NSModel<SortPolicy>::Naive() const
{
  if (nSearch)
    return nSearch->Naive();
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::Naive()
{
  if (nSearch)
    return nSearch->Naive();
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool NSModel<SortPolicy>::SingleMode() const
{
  if (nSearch)
    return nSearch->SingleMode();
  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::SingleMode()
{
  if (nSearch)
    return nSearch->SingleMode();
  throw std::runtime_error("no neighbor search model initialized");
}

//! Get the name of the tree type.
template<typename SortPolicy>
std::string NSModel<SortPolicy>::TreeName() const
{
  switch (treeType)
  {
    case KD_TREE:
      return "kd-tree";
    case COVER_TREE:
      return "cover tree";
    case R_TREE:
      return "R tree";
    case R_STAR_TREE:
      return "R* tree";
    case BALL_TREE:
      return "ball tree";
    case X_TREE:
      return "X tree";
    default:
      return "unknown tree";
  }
}

//! Serialize the kNN model.
template<typename SortPolicy>
template<typename Archive>
void NSModel<SortPolicy>::Serialize(Archive& ar,
                                    const unsigned int /* version */)
{
  ar & data::CreateNVP(treeType, "treeType");
  ar & data::CreateNVP(randomBasis, "randomBasis");
  ar & data::CreateNVP(q, "q");

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
  {
    if (nSearch)
      delete nSearch;

    nSearch = NULL;
  }
  // We'll only need to serialize one of the kNN objects, based on the type.
  // TODO This doesn't work... should be fixed.
  const std::string& name = NSModelName<SortPolicy>::Name();
  switch (treeType)
  {
    case KD_TREE:
    {
      NSLeaf<tree::KDTree>* childPtr = (NSLeaf<tree::KDTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
    case COVER_TREE:
    {
      NSType<tree::StandardCoverTree>* childPtr =
          (NSType<tree::StandardCoverTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
    case R_TREE:
    {
      NSType<tree::RTree>* childPtr = (NSType<tree::RTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
    case R_STAR_TREE:
    {
      NSType<tree::RStarTree>* childPtr = (NSType<tree::RStarTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
    case BALL_TREE:
    {
      NSLeaf<tree::BallTree>* childPtr = (NSLeaf<tree::BallTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
    case X_TREE:
    {
      NSType<tree::XTree>* childPtr = (NSType<tree::XTree>*) nSearch;
      ar & data::CreateNVP(childPtr, name);
      nSearch = childPtr;
      break;
    }
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
