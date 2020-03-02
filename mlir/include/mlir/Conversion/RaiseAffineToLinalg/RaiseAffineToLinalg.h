#ifndef MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_H
#define MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_H

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {

namespace {

enum class FUNCTION {
  MATMUL,
  RESHAPE,
  TRANSPOSE,
};

std::string composeFunctionNameForMatmul(const ArrayRef<Type> &types) {
  if (types.size() != 3)
    llvm_unreachable("expect 3 memref");
  auto AShape = types[0].dyn_cast<MemRefType>().getShape();
  auto CShape = types[2].dyn_cast<MemRefType>().getShape();
  std::string result = "Matmul_";
  result += std::to_string(CShape[0]) + "x" + std::to_string(CShape[1]) + "x" +
            std::to_string(AShape[1]);
  return result;
}

std::string composeFunctionNameForReshape(const ArrayRef<Type> &types) {
  if (types.size() != 2)
    llvm_unreachable("expect single memref");
  std::string result = "Reshape_";
  auto SShape = types[0].dyn_cast<MemRefType>().getShape();
  auto DShape = types[1].dyn_cast<MemRefType>().getShape();
  for (size_t i = 0; i < SShape.size() - 1; i++)
    result += std::to_string(SShape[i]) + "x";
  result += std::to_string(SShape[SShape.size() - 1]);
  result += "_to_";
  for (size_t i = 0; i < DShape.size() - 1; i++)
    result += std::to_string(DShape[i]) + "x";
  result += std::to_string(DShape[DShape.size() - 1]);
  return result;
}

std::string composeFunctionNameForTranspose(const ArrayRef<Type> &types) {
  std::string result = "Transpose_";
  auto TShape = types[0].dyn_cast<MemRefType>().getShape();
  for (size_t i = 0; i < TShape.size() - 1; i++)
    result += std::to_string(TShape[i]) + "x";
  result += std::to_string(TShape[TShape.size() - 1]);
  return result;
}

// TODO: enforce pre-conditions.
template <typename... Args>
std::string composeFunctionCallName(FUNCTION id, const Args... args) {
  ArrayRef<Type> types = {args...};
  switch (id) {
  case FUNCTION::MATMUL:
    return composeFunctionNameForMatmul(types);
  case FUNCTION::RESHAPE:
    return composeFunctionNameForReshape(types);
  case FUNCTION::TRANSPOSE:
    return composeFunctionNameForTranspose(types);
  }
  return "nullptr";
}

} // end namespace

} // end namespace mlir.

#endif
