// src/split/src/cvec_load.cpp - see cvec_load.hpp. Mirrors the main engine's load_control_vectors (its file loop,
// the project-mode normalization over doubles, the layer defaults) on fnx::Gguf; the tables and the per-request flag
// live in strata::kernels (cvec_upload / cvec_replicate), and the engine applies them where a steered layer's FFN
// write would otherwise fold into the next read (src/engine.cu, src/prefill.cu).
#include "cvec_load.hpp"

#include "gguf.hpp"
#include "model.hpp"

#include "strata/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fnx {

bool load_control_vectors(const std::vector<std::pair<std::string, float>>& files, int mode, int first_in, int last_in,
                          int single_in, std::string& summary, std::string& err) {
    const int64_t L = NL, N = ::fnx::N;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : files) {
        try {
            Gguf f(path);
            if (!f.has_key("general.architecture") || f.s("general.architecture") != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            if (f.has_key("controlvector.model_hint") && f.s("controlvector.model_hint") != "qwen4exp")
                std::fprintf(stderr, "sfnx: %s was made for '%s', not qwen4exp\n", path.c_str(),
                             f.s("controlvector.model_hint").c_str());
            int found = 0;
            for (const auto& kv : f.tensors()) {
                const Tensor& t = kv.second;
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != T_F32 || t.ne[0] != N || t.nrows() != 1) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(t.data);
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = first_in <= 0 ? 1 : first_in;
    const int last = (last_in <= 0 || last_in >= L) ? (int) L - 1 : last_in;
    const int single = mode == 0 ? single_in : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::memcpy(dir.data() + (size_t) (l * N), d, (size_t) N * sizeof(float));
        }
        ++steered;
    }
    if (steered == 0) {
        err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last);
        return false;
    }
    if (!strata::kernels::cvec_upload(dir, s, mode, first, last, N, HC, err)) return false;
    summary = std::string(mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "sfnx: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

bool cvec_place(const std::vector<int>& devs, std::string& err) {
    if (!strata::kernels::cvec().loaded()) return true;   // no vector: nothing to place
    int prev = 0;
    cudaGetDevice(&prev);
    for (const int d : devs) {
        if (cudaSetDevice(d) != cudaSuccess) {
            cudaGetDevice(&prev);
            err = "control vector: cannot switch to card device " + std::to_string(d);
            return false;
        }
        if (!strata::kernels::cvec_replicate(err)) { cudaSetDevice(prev); return false; }
    }
    cudaSetDevice(prev);
    return true;
}

}  // namespace fnx
