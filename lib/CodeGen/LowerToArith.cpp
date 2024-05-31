#include "filskalang/CodeGen/LowerToArith.h"
#include "filskalang/CodeGen/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {
struct NegOpLowering : public mlir::ConversionPattern {
public:
  explicit NegOpLowering(mlir::MLIRContext *Context)
      : ConversionPattern(mlir::filskalang::NegOp::getOperationName(), 1,
                          Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op, mlir::ArrayRef<mlir::Value> Operands,
                  mlir::ConversionPatternRewriter &Rewriter) const override {
    auto Loc = Op->getLoc();
    auto NegOp = mlir::cast<mlir::filskalang::NegOp>(Op);

    auto NegFOp = Rewriter.create<mlir::arith::NegFOp>(Loc, NegOp.getArg());

    Rewriter.replaceOp(Op, NegFOp);
    return mlir::success();
  }
};

struct FilskalangToArithLoweringPass
    : public mlir::PassWrapper<FilskalangToArithLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FilskalangToArithLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void FilskalangToArithLoweringPass::runOnOperation() {
  // define final target
  mlir::ConversionTarget Target(getContext());

  Target.addLegalDialect<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                         mlir::filskalang::FilskalangDialect>();
  Target.addIllegalOp<mlir::filskalang::NegOp>();

  mlir::RewritePatternSet Patterns(&getContext());

  Patterns.add<NegOpLowering>(&getContext());

  auto Module = getOperation();
  if (failed(applyPartialConversion(Module, Target, std::move(Patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::filskalang::createLowerToArithPass() {
  return std::make_unique<FilskalangToArithLoweringPass>();
}
