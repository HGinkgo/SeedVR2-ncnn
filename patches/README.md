# PNNX PT2 Patch

`pnnx-pt2-independent.patch` targets ncnn commit `c6b351b56fbe32e0381ae00331e3df649b20d7b7`. It adds a native ExportedProgram reader and two basic PT2 tests without changing the SeedVR2 ncnn submodule.

Apply and test it with:

```bash
tools/prepare_pnnx_pt2.sh --python /path/to/python --jobs 4
```

The patch supports the fixed operator subset needed by the project, including graph and weight loading, input/output signatures, Linear, RMSNorm, SiLU, elementwise operations, reshape/slice/select/permute, matrix multiplication, attention, and the fixed VAE subset. Unsupported graph features fail explicitly.

This is an independent project patch. It does not claim complete PT2 operator coverage or replace the production TorchScript/ncnn model package.
