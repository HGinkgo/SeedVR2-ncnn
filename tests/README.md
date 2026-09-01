# Test Tiers

The test directory is intentionally split by dependency and runtime cost:

- `unit/cpp/`: fast native contracts and custom-layer checks. These are registered with CTest; Vulkan builds add the backend-contract cases.
- `integration/cpp/`: model-backed graph-load, golden, and end-to-end checks. They are opt-in because they require model files, reference records, and a suitable runtime.
- `python/`: exporter, conditioning, dynamic-package materialization, and reference-tool checks. They do not run as part of the native CTest suite.
- `manual/package/`: release archive validation for Linux and Windows runtime packages.
- `manual/cpp/`: opt-in AWA benchmark tooling, separate from correctness tests.
- `smoke.sh`: the dependency-light CLI smoke test used by the CPU CI job.

There are no duplicate correctness tests in the current tree: similarly named integration programs exercise different graph boundaries or execution backends. Generated Python caches are ignored and are not release inputs.
