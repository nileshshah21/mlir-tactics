#include "Lib/access_patterns.h"
#include "Lib/builders.h"
#include "Lib/ctx.h"
#include "Lib/islAst.h"
#include "Lib/islNodeBuilder.h"
#include "Lib/matchers.h"
#include "Lib/mlirCodegen.h"
#include "Lib/scop.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>
#include <iostream>

#define DEBUG_TYPE "pet-to-mlir"

using namespace llvm;

static cl::OptionCategory toolOptions("pet to mlir - tool options");

static cl::opt<std::string> outputFileName("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("out"),
                                           cl::cat(toolOptions));

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<Specify input file>"),
                                          cl::Required, cl::cat(toolOptions));

static cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> reschedule("reschedule",
                                llvm::cl::desc("Reschedule with ISL"),
                                llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> dumpSchedule("dump-schedule",
                                  llvm::cl::desc("Pretty print the schedule"),
                                  llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> dumpScop("dump-scop",
                              llvm::cl::desc("Pretty print the scop"),
                              llvm::cl::init(false), cl::cat(toolOptions));

static cl::opt<bool> dumpAst("dump-ast", llvm::cl::desc("Pretty print the ast"),
                             llvm::cl::init(false), cl::cat(toolOptions));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::opt<bool> enableLoopTactics("enable-loop-tactics",
                                       llvm::cl::desc("Enable loop tactics"),
                                       llvm::cl::init(false),
                                       cl::cat(toolOptions));

// check if the schedule is bounded.
static bool isUnbounded(isl::schedule schedule) {
  auto isUnBoundedSet = [](isl::set set) -> bool {
    return !(set.is_bounded());
  };
  std::vector<isl::set> domains;
  schedule.get_domain().foreach_set([&](isl::set set) {
    domains.push_back(set);
    return isl_stat_ok;
  });
  int unBounded = count_if(domains.begin(), domains.end(), isUnBoundedSet);
  return unBounded != 0;
}

static void dumpScheduleWithIsl(isl::schedule schedule, llvm::raw_ostream &os) {
  auto ctx = schedule.get_ctx().get();
  auto *p = isl_printer_to_str(ctx);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule(p, schedule.get());
  auto *str = isl_printer_get_str(p);
  os << str << "\n";
  free(str);
  isl_printer_free(p);
}

static isl::schedule rescheduleWithIsl(pet::Scop &scop) {
  auto proximity = scop.getAllDependences();
  auto validity = scop.getAllDependences();

  auto sc = isl::schedule_constraints::on_domain(scop.getNonKilledDomain());

  sc = sc.set_proximity(proximity);
  sc = sc.set_validity(validity);
  sc = sc.set_coincidence(validity);
  auto schedule = sc.compute_schedule();
  return schedule;
}

static inline isl::schedule_node
rebuild(isl::schedule_node node,
        const builders::ScheduleNodeBuilder &replacement) {
  // this may not be always legal...
  node = node.cut();
  node = replacement.insertAt(node);
  return node;
}

static isl::schedule_node
replaceOnce(isl::schedule_node node,
            const matchers::ScheduleNodeMatcher &pattern,
            const builders::ScheduleNodeBuilder &replacement) {
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    LLVM_DEBUG(dbgs() << "match success!\n");
    node = rebuild(node, replacement);
  }
  return node;
}

static isl::schedule_node
replaceDFSPreorderOnce(isl::schedule_node node,
                       const matchers::ScheduleNodeMatcher &pattern,
                       const builders::ScheduleNodeBuilder &replacement) {
  node = replaceOnce(node, pattern, replacement);
  if ((isl_schedule_node_get_type(node.get())) == isl_schedule_node_mark) {
    return node;
  }
  for (int i = 0; i < node.n_children(); ++i) {
    node = replaceDFSPreorderOnce(node.child(i), pattern, replacement).parent();
  }
  return node;
}

static isl::schedule runDetection(pet::Scop &scop) {
  auto root = scop.getSchedule().get_root();

  isl::schedule_node ijk;

  using namespace matchers;

  // check if the partial schedule is 3d.
  auto is3d = [](isl::schedule_node band) {
    auto umap = band.child(0).get_prefix_schedule_union_map();
    if (umap.n_map() != 1)
      return false;
    auto map = isl::map::from_union_map(umap);
    return map.dim(isl::dim::out) == 3;
  };

  auto hasGemmAccess = [&scop](isl::schedule_node band) {
    auto reads = scop.getReads();
    auto writes = scop.getMustWrites();
    reads = reads.apply_domain(band.child(0).get_prefix_schedule_union_map());
    writes = writes.apply_domain(band.child(0).get_prefix_schedule_union_map());

    using namespace matchers;
    auto ctx = band.get_ctx();
    auto _i = placeholder(ctx);
    auto _ii = placeholder(ctx);
    auto _j = placeholder(ctx);
    auto _jj = placeholder(ctx);
    auto _k = placeholder(ctx);
    auto _A = arrayPlaceholder();
    auto _B = arrayPlaceholder();
    auto _C = arrayPlaceholder();

    // placeholder are *not* reused across different calls of allOf.
    auto psRead =
        allOf(access(_C, _i, _j), access(_A, _i, _k), access(_B, _k, _j));
    auto psWrite = allOf(access(_C, _ii, _jj));
    auto readMatches = match(reads, psRead);
    auto writeMatches = match(writes, psWrite);

    if ((readMatches.size() != 1) || (writeMatches.size() != 1))
      return false;

    if ((readMatches[0][_i].payload().inputDimPos_ !=
         writeMatches[0][_ii].payload().inputDimPos_) ||
        (readMatches[0][_j].payload().inputDimPos_ !=
         writeMatches[0][_jj].payload().inputDimPos_))
      return false;
    return true;
  };

  auto isGemmLike = [&](isl::schedule_node band) {
    return is3d(band) && hasGemmAccess(band);
  };

  // clang-format off
  auto matcher =
  band(isGemmLike, ijk, 
    leaf());
  // clang-format on

  auto builder = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    auto scheduleIJK = [&]() { return ijk.band_get_partial_schedule(); };
    auto marker = [&]() {
      return isl::id::alloc(ijk.get_ctx(), "MatMul", nullptr);
    };
    // clang-format off
    builder = mark(marker, band(scheduleIJK));
    // clang-format on
  }

  root = replaceDFSPreorderOnce(root, matcher, builder);
  return root.get_schedule();
}

int main(int argc, char **argv) {

  using namespace mlir;
  using namespace pet;
  using namespace util;
  using namespace ast;
  using namespace codegen;

  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream inputFile(inputFileName);
  if (!inputFile.good()) {
    outs() << "Not able to open file: " << inputFileName << "\n";
    return -1;
  }

  // pass include paths to pet.
  struct pet_options *options;
  options = pet_options_new_with_defaults();
  std::vector<char *> arguments;
  char argument1[] = "program";
  char argumentI[] = "-I";
  arguments.push_back(argument1);
  for (const auto &includePath : includeDirs) {
    arguments.push_back(argumentI);
    arguments.push_back(const_cast<char *>(includePath.c_str()));
  }
  int argsCount = arguments.size();
  argsCount = pet_options_parse(options, argsCount, &arguments[0], ISL_ARG_ALL);
  auto ctx = ScopedCtx(isl_ctx_alloc_with_options(&pet_options_args, options));

  auto petScop = Scop::parseFile(ctx, inputFileName);
  if (!petScop.isValid()) {
    outs() << "Invalid scop\n";
    return -1;
  }

  if (dumpScop)
    petScop.dump();

  if (dumpAst)
    IslAst(petScop).dump();

  // bail-out if we have symbolic constants.
  auto contextSet = petScop.getContext();
  auto params = contextSet.get_space().dim(isl::dim::param);
  if (params > 0) {
    outs() << "we do not allow symbolic constant at the moment."
           << "\n";
    return -1;
  }

  // bail-out if the schedule domain is unbounded.
  if (isUnbounded(petScop.getSchedule())) {
    outs() << "schedule must be bounded\n";
    return -1;
  }

  if (reschedule) {
    auto newSchedule = rescheduleWithIsl(petScop);
    if (!newSchedule) {
      outs() << "failed to reschedule\n";
      return -1;
    }
    petScop.schedule() = newSchedule;
  }

  if (enableLoopTactics) {
    if (!reschedule) {
      outs() << "use loop tactics with the reschedule option\n";
      return -1;
    }
    petScop.schedule() = runDetection(petScop);
  }

  if (dumpSchedule)
    dumpScheduleWithIsl(petScop.getSchedule(), outs());

  registerDialect<AffineDialect>();
  registerDialect<StandardOpsDialect>();
  MLIRContext context;
  if (showDialects) {
    outs() << "Registered Dialects:\n";
    for (Dialect *dialect : context.getRegisteredDialects()) {
      outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }
  MLIRCodegen MLIRbuilder(context, petScop);

  auto ISLAst = IslAst(petScop);
  // ISLAst.dump();

  auto ISLNodeBuilder = IslNodeBuilder(ISLAst, MLIRbuilder);
  ISLNodeBuilder.MLIRFromISLAst();
  // MLIRbuilder.dump();

  if (outputFileName.empty()) {
    MLIRbuilder.print(outs());
    return 0;
  }

  std::error_code ec;
  ToolOutputFile out(outputFileName, ec, sys::fs::OF_None);
  if (ec) {
    WithColor::error() << ec.message() << "\n";
    return -1;
  }
  MLIRbuilder.print(out.os());
  out.keep();
  return 0;
}
