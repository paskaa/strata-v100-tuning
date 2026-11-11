// Control vectors on the split engine's residual stream: the main engine's loader (src/program/generate.cpp's
// load_control_vectors) with the same semantics, reading the `controlvector` GGUFs with the split's own Gguf reader
// and uploading the tables through strata::kernels::cvec_upload (src/kernels/cuda/cvec.cu, already linked from
// libstrata_kernels.a). See include/strata/kernels/cvec.hpp for what the tables mean and where the vector applies.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace fnx {

/// Files: (path, scale), every file's `direction.<layer>` f32 rows summed times its scale (layer 0 has none).
/// `mode` 0 = project (unit direction, its norm as the scale), 1 = add. `first`/`last` are the steered layers
/// inclusive (<= 0: the defaults - 1 .. the last layer); `single` >= 0 (project only): that layer's direction for
/// every steered layer. Reads with fnx::NL layers and fnx::N as wide as the model. `summary` is what sfnx's INFO
/// line reports as cvec= ("project:4-44"). false with `err` set: the caller must not start with half a vector.
bool load_control_vectors(const std::vector<std::pair<std::string, float>>& files, int mode, int first, int last,
                          int single, std::string& summary, std::string& err);

/// Put the loaded vector's tables on every card it runs on: a layer split applies the vector on all of them
/// (cvec_apply uses the tables of the device it runs on). No-op without a vector. Uploads on the CURRENT device
/// first, so call this once after load_control_vectors with the model's devices listed.
bool cvec_place(const std::vector<int>& devs, std::string& err);

}  // namespace fnx
