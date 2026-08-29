# Local alias so the Coral docs' `//sim:coralnpu_v2_sim` label works from this
# (coralnpu_hw) repo. The real CLI binary lives in the external MPACT-CoralNPU
# repo (@coralnpu_mpact). Build/run it with the npusim config, which supplies the
# clang platform and execroot include path the MPACT sources need:
#
#   bazel run --config=npusim //sim:coralnpu_v2_sim -- <program.elf>
#
package(default_visibility = ["//visibility:public"])

alias(
    name = "coralnpu_v2_sim",
    actual = "@coralnpu_mpact//sim:coralnpu_v2_sim",
)
