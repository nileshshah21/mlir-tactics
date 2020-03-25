#ifndef HELPERS_GENERATION_TACTICS_GPU
#define HELPERS_GENERATION_TACTICS_GPU

#include "HelpersCommon.h"

/*
  Helpers for tablegen backend code generation - gpu path
*/

namespace {

// get the size of the memref.
// TODO: enfore all the dimension to be known at compile time.
// move this in the runtime library?
// TODO: elementSize is assumed to be float (4 bytes).
int64_t getSizeBuffer(mlir::Type bufferType) {
  auto memref = bufferType.dyn_cast<mlir::MemRefType>();
  auto memrefShape = memref.getShape();
  int64_t totalSize = 1;
  for (auto dim : memrefShape)
    totalSize *= dim;
  auto elemSize = 4;
  totalSize *= elemSize;
  return totalSize;
}

// create a function to allocate memory on the device.
// The created function looks like:
//
// %0 = call @allocateMemoryForDevice(%c4194304_i64) : (i64) -> !llvm<"i8*">
//
// Arg1 is the size of the buffer that needs to be allocated on the device.
// Ret is the pointer to such buffer.
mlir::Value createCallAllocateMemoryForDevice(mlir::ModuleOp module,
                                              mlir::PatternRewriter &rewriter,
                                              mlir::Location loc,
                                              int64_t size) {
  std::string name = "allocateMemoryForDevice";
  mlir::Type i64Type = rewriter.getIntegerType(64);
  auto *llvmDialect = getLLVMDialect(module);
  mlir::FlatSymbolRefAttr symbolFn =
      getOrInsertFunction(rewriter, module, name, i64Type,
                          mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect));
  mlir::Value operand = rewriter.create<mlir::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(size));
  return rewriter
      .create<mlir::CallOp>(
          loc, symbolFn,
          llvm::ArrayRef<mlir::Type>{
              mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect)},
          operand)
      .getResult(0);
}

// create a function to move data from host to device.
// The created function looks like:
//
// call @createCallCopyFromHostToDevice(%arg2, %0, %c4194304_i64) :
//    (memref<1024x1024xf32>, !llvm<"i8*">, i64) -> ()
//
// Arg1 is the memref (source).
// Arg2 is the pointer to the device buffer (devA - destination).
// Arg3 is the number of bytes to be moved.
// Ret void
void createCallCopyFromHostToDevice(mlir::ModuleOp module,
                                    mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value source,
                                    mlir::Value dest, int64_t size) {
  std::string name = "createCallCopyFromHostToDevice";
  auto *llvmDialect = getLLVMDialect(module);
  auto pointerType = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  mlir::Type i64Type = rewriter.getIntegerType(64);
  mlir::FlatSymbolRefAttr symbolFn = getOrInsertFunction(
      rewriter, module, name, {source.getType(), pointerType, i64Type});
  mlir::Value sizeOp = rewriter.create<mlir::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(size));
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{source, dest, sizeOp});
}

// create a function to execut cublasSgemm.
// The created function looks like:
//
// @createCallToCublasSgemm(%0, %1, %2, %arg2, %arg0, %arg1) :
// (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">,
//  memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()
//
// Arg1 is the pointer to buffer devC.
// Arg2 is the pointer to buffer devA.
// Arg3 is the pointer to buffer devB.
// Arg4 is the memref C (we pass memref to get leading dimension/M,N and K).
// Arg5 is the memref A.
// Arg6 is the memref B.
// Ret void
void createCallToCublasSgemm(mlir::ModuleOp module,
                             mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value devC,
                             mlir::Value devA, mlir::Value devB, mlir::Value C,
                             mlir::Value A, mlir::Value B) {
  std::string name = "createCallToCublasSgemm";
  auto *llvmDialect = getLLVMDialect(module);
  auto pointerType = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  mlir::FlatSymbolRefAttr symbolFn =
      getOrInsertFunction(rewriter, module, name,
                          {pointerType, pointerType, pointerType, C.getType(),
                           A.getType(), B.getType()});
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{devC, devA, devB, C, A, B});
}

// create function a function to move data from device to host.
// The created function looks like:
//
// call @createCallCopyFromDeviceToHost(%0, %arg2, %c4194304_i64) :
// (!llvm<"i8*">, memref<1024x1024xf32>, i64) -> ()
//
// Arg1 is the pointer to the device buffer (source).
// Arg2 is the memref (destination).
// Arg3 is the number of bytes to be moved.
void createCallCopyFromDeviceToHost(mlir::ModuleOp module,
                                    mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value source,
                                    mlir::Value dest, int64_t size) {
  std::string name = "createCallCopyFromDeviceToHost";
  auto *llvmDialect = getLLVMDialect(module);
  auto pointerType = mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  mlir::Type i64Type = rewriter.getIntegerType(64);
  mlir::FlatSymbolRefAttr symbolFn = getOrInsertFunction(
      rewriter, module, name, {pointerType, dest.getType(), i64Type});
  mlir::Value sizeOp = rewriter.create<mlir::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(size));
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{source, dest, sizeOp});
}

} // end namespace

#endif
