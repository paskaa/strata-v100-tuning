// Host-only test of the control-vector loader (src/cvec_load.cpp): writes tiny `controlvector` GGUFs, runs
// fnx::load_control_vectors on them with strata::kernels::cvec_upload stubbed out (the tables are captured, no GPU),
// and checks the project/add math, the file-scale accumulation, single:L, the layer range, the summary string and
// every error path. cvec_place is not covered: it only moves tables between cards (cudaSetDevice + cvec_replicate).
// Build: cmake --build <build> --target split_cvec_load_test; run: build/.../src/split/split_cvec_load_test [DIR].
#include "../src/cvec_load.hpp"
#include "../src/model.hpp"

#include "strata/kernels/cvec.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// The kernel upload, stubbed: the loader's tables land here instead of on a card.
namespace strata::kernels {
namespace {
struct Cap {
    std::vector<float> dir, s;
    int mode = -1, first = -1, last = -1;
    int64_t n_embd = 0, hc = 0;
    int calls = 0;
};
Cap g_cap;
}  // namespace
bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    (void) err;
    g_cap.dir = dir;
    g_cap.s = s;
    g_cap.mode = mode;
    g_cap.first = first;
    g_cap.last = last;
    g_cap.n_embd = n_embd;
    g_cap.hc = hc;
    ++g_cap.calls;
    return true;
}
const Cvec& cvec() {
    static Cvec none;
    return none;
}
bool cvec_replicate(std::string& err) {
    (void) err;
    return true;
}
}  // namespace strata::kernels

namespace {

const int64_t N = fnx::N, L = fnx::NL;   // 2560, 48: every direction row is this wide, layers 1..47 steer

int failures = 0;
#define CHECK(cond, tag, fmt, ...)                                                       \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::printf("FAIL %s: " fmt "\n", tag, ##__VA_ARGS__);                        \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)
#define CHECK_OK(cond, tag)                                                              \
    do {                                                                                 \
        if (cond) {                                                                      \
            std::printf("ok   %s\n", tag);                                                \
        } else {                                                                         \
            std::printf("FAIL %s\n", tag);                                               \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)

bool near(float got, double want, double eps = 1e-4) {
    const double d = std::fabs((double) got - want);
    return d <= eps * (1.0 + std::fabs(want));
}

// One F32 tensor's bytes.
struct Ten {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    std::vector<uint8_t> raw;
};

Ten f32dir(int layer, const std::vector<float>& row) {
    Ten t;
    t.name = "direction." + std::to_string(layer);
    t.type = 0;
    t.ne = {(uint64_t) row.size()};
    t.raw.resize(row.size() * 4);
    std::memcpy(t.raw.data(), row.data(), t.raw.size());
    return t;
}

std::vector<float> const_row(float c) { return std::vector<float>((size_t) N, c); }

std::vector<float> ramp_row() {
    std::vector<float> r((size_t) N);
    for (int64_t j = 0; j < N; ++j) r[(size_t) j] = (float) (j + 1);
    return r;
}

// A minimal GGUF with `arch` and the given tensors (only string type 8 keys, F32/F16 data).
std::vector<uint8_t> make(const std::string& arch, const std::vector<Ten>& ts, const std::string& hint = {}) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    auto str = [&](const std::string& s) {
        u64(s.size());
        put(s.data(), s.size());
    };
    u32(0x46554747u);   // "GGUF"
    u32(3);             // version
    u64(ts.size());
    u64(hint.empty() ? 1 : 2);
    str("general.architecture");
    u32(8);
    str(arch);
    if (!hint.empty()) {
        str("controlvector.model_hint");
        u32(8);
        str(hint);
    }
    uint64_t off = 0;
    for (const Ten& t : ts) {
        str(t.name);
        u32((uint32_t) t.ne.size());
        for (uint64_t v : t.ne) u64(v);
        u32(t.type);
        u64(off);
        off += t.raw.size();
    }
    while (b.size() % 32) b.push_back(0);
    for (const Ten& t : ts) put(t.raw.data(), t.raw.size());
    return b;
}

std::string write(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b) {
    const std::string path = dir + "/fnx_cvec_load_test_" + tag + ".gguf";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size() || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
    return path;
}

void reset_cap() { strata::kernels::g_cap = strata::kernels::Cap(); }

bool run_ok(const std::string& tag, const std::vector<std::pair<std::string, float>>& files, int mode, int first,
            int last, int single, std::string& summary, std::string& err) {
    reset_cap();
    summary.clear();
    err.clear();
    // the loader prints the llama.cpp-style line to stderr; the test only reads the return value
    const bool ok = fnx::load_control_vectors(files, mode, first, last, single, summary, err);
    if (!ok) {
        std::printf("FAIL %s: %s\n", tag.c_str(), err.c_str());
        ++failures;
        return false;
    }
    return true;
}

void run_bad(const std::string& tag, const std::vector<std::pair<std::string, float>>& files, int mode, int first,
             int last, int single, const char* want) {
    reset_cap();
    std::string summary, err;
    // keep stderr quiet here would hide nothing: a refused load prints no summary line
    const bool ok = fnx::load_control_vectors(files, mode, first, last, single, summary, err);
    const bool good = !ok && std::strstr(err.c_str(), want) && strata::kernels::g_cap.calls == 0;
    std::printf("%s %s: %s\n", good ? "ok  " : "FAIL", tag.c_str(), ok ? "loaded" : err.c_str());
    failures += !good;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";
    const double root_n = std::sqrt((double) N);

    {   // project, one layer: the unit direction and its norm, the summary, the upload's geometry
        const std::string p = write(dir, "proj1", make("controlvector", {f32dir(7, const_row(2.0f))}));
        std::string summary, err;
        if (run_ok("project_one", {{p, 1.0f}}, 0, 0, 0, -1, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            bool ok = summary == "project:1-47" && c.mode == 0 && c.first == 1 && c.last == 47 &&
                      c.n_embd == N && c.hc == fnx::HC && c.calls == 1 && c.s.size() == (size_t) L &&
                      c.dir.size() == (size_t) (L * N);
            ok = ok && near(c.s[7], 2.0 * root_n) && near(c.dir[(size_t) (7 * N)], 1.0 / root_n) &&
                 near(c.dir[(size_t) (7 * N + N - 1)], 1.0 / root_n);
            for (int64_t l = 1; l < L && ok; ++l) {
                if (l == 7) continue;
                ok = c.s[(size_t) l] == 0.0f && c.dir[(size_t) (l * N)] == 0.0f;
            }
            CHECK_OK(ok, "project_one");
            if (!ok)
                std::printf("     summary '%s' s7 %f dir7 %f\n", summary.c_str(), c.s[7],
                            c.dir[(size_t) (7 * N)]);
        }
        std::remove(p.c_str());
    }

    {   // project over a ramp: the norm is over doubles, each entry divided by it
        const std::string p = write(dir, "ramp", make("controlvector", {f32dir(7, ramp_row())}));
        std::string summary, err;
        if (run_ok("project_ramp", {{p, 1.0f}}, 0, 0, 0, -1, summary, err)) {
            double nrm = 0.0;
            for (int64_t j = 1; j <= N; ++j) nrm += (double) j * j;
            nrm = std::sqrt(nrm);
            const auto& c = strata::kernels::g_cap;
            const bool ok = near(c.s[7], nrm) && near(c.dir[(size_t) (7 * N)], 1.0 / nrm) &&
                            near(c.dir[(size_t) (7 * N + N - 1)], (double) N / nrm);
            CHECK_OK(ok, "project_ramp");
        }
        std::remove(p.c_str());
    }

    {   // two files, scales accumulate per layer before the norm; a foreign model hint only warns
        const std::string a = write(dir, "two_a", make("controlvector", {f32dir(5, const_row(1.0f))}));
        const std::string b =
            write(dir, "two_b", make("controlvector", {f32dir(5, const_row(1.0f)), f32dir(6, const_row(4.0f))}, "other"));
        std::string summary, err;
        if (run_ok("two_files_scaled", {{a, 2.0f}, {b, 0.5f}}, 0, 0, 0, -1, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            const bool ok = summary == "project:1-47" && near(c.s[5], 2.5 * root_n) &&
                            near(c.dir[(size_t) (5 * N)], 1.0 / root_n) && near(c.s[6], 2.0 * root_n);
            CHECK_OK(ok, "two_files_scaled");
        }
        std::remove(a.c_str());
        std::remove(b.c_str());
    }

    {   // add: no norm, the scaled sum lands as-is, every steered layer's scale is 1
        const std::string p =
            write(dir, "add", make("controlvector", {f32dir(5, const_row(1.0f)), f32dir(6, ramp_row())}));
        std::string summary, err;
        if (run_ok("add", {{p, 2.0f}}, 1, 0, 0, -1, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            bool ok = summary == "add:1-47" && c.mode == 1 && c.s[5] == 1.0f && c.s[6] == 1.0f &&
                      c.s[4] == 0.0f && near(c.dir[(size_t) (5 * N)], 2.0) &&
                      near(c.dir[(size_t) (6 * N + N - 1)], (double) N * 2.0);
            CHECK_OK(ok, "add");
        }
        std::remove(p.c_str());
    }

    {   // add ignores single:L (per-layer anyway), and the summary carries no :single suffix
        const std::string p = write(dir, "add_single", make("controlvector", {f32dir(5, const_row(1.0f))}));
        std::string summary, err;
        if (run_ok("add_ignores_single", {{p, 1.0f}}, 1, 0, 0, 5, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            CHECK_OK(summary == "add:1-47" && near(c.dir[(size_t) (5 * N)], 1.0), "add_ignores_single");
        }
        std::remove(p.c_str());
    }

    {   // single:L (project): every steered layer shares layer 5's unit direction and norm, not its own
        const std::string p =
            write(dir, "single", make("controlvector", {f32dir(5, const_row(2.0f)), f32dir(9, const_row(4.0f))}));
        std::string summary, err;
        if (run_ok("single", {{p, 1.0f}}, 0, 0, 0, 5, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            bool ok = summary == "project:1-47:single5";
            for (int64_t l = 1; l < L && ok; ++l)
                ok = near(c.s[(size_t) l], 2.0 * root_n) && near(c.dir[(size_t) (l * N)], 1.0 / root_n);
            CHECK_OK(ok, "single");
        }
        std::remove(p.c_str());
    }

    {   // layer range: directions outside first..last stay unsteered even when the file has them
        std::vector<Ten> ts;
        for (int l = 2; l <= 8; ++l) ts.push_back(f32dir(l, const_row(1.0f)));
        const std::string p = write(dir, "range", make("controlvector", ts));
        std::string summary, err;
        if (run_ok("range", {{p, 1.0f}}, 0, 4, 6, -1, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            bool ok = summary == "project:4-6" && c.first == 4 && c.last == 6;
            for (int64_t l = 1; l < L && ok; ++l) {
                const bool in = l >= 4 && l <= 6;
                ok = in ? near(c.s[(size_t) l], root_n) : c.s[(size_t) l] == 0.0f;
            }
            CHECK_OK(ok, "range");
        }
        std::remove(p.c_str());
    }

    {   // a range past the model clamps to the last layer (47), like the main engine's loader
        const std::string p = write(dir, "clamp", make("controlvector", {f32dir(47, const_row(1.0f))}));
        std::string summary, err;
        if (run_ok("range_clamp", {{p, 1.0f}}, 0, 40, 1000, -1, summary, err)) {
            const auto& c = strata::kernels::g_cap;
            CHECK_OK(summary == "project:40-47" && c.last == 47 && near(c.s[47], root_n), "range_clamp");
        }
        std::remove(p.c_str());
    }

    // Every refusal: no upload happens (calls stays 0).
    {
        const std::string p = write(dir, "bad_arch", make("llama", {f32dir(5, const_row(1.0f))}));
        run_bad("bad_arch", {{p, 1.0f}}, 0, 0, 0, -1, "not a control vector");
        std::remove(p.c_str());
    }
    {
        Ten w;
        w.name = "weight";
        w.type = 0;
        w.ne = {(uint64_t) N};
        w.raw.assign((size_t) N * 4, 0);
        const std::string p = write(dir, "no_dirs", make("controlvector", {w}));
        run_bad("no_dirs", {{p, 1.0f}}, 0, 0, 0, -1, "no direction");
        std::remove(p.c_str());
    }
    {
        const std::string p = write(dir, "layer0", make("controlvector", {f32dir(0, const_row(1.0f))}));
        run_bad("layer0_ignored", {{p, 1.0f}}, 0, 0, 0, -1, "no direction");
        std::remove(p.c_str());
    }
    {
        const std::string p = write(dir, "layer49", make("controlvector", {f32dir(49, const_row(1.0f))}));
        run_bad("layer_past_model_ignored", {{p, 1.0f}}, 0, 0, 0, -1, "no direction");
        std::remove(p.c_str());
    }
    {
        Ten t = f32dir(5, const_row(1.0f));
        t.ne = {1280};
        t.raw.resize(1280 * 4);
        const std::string p = write(dir, "wide", make("controlvector", {t}));
        run_bad("bad_width", {{p, 1.0f}}, 0, 0, 0, -1, "must be 2560 f32");
        std::remove(p.c_str());
    }
    {
        Ten t;
        t.name = "direction.5";
        t.type = 1;   // F16
        t.ne = {(uint64_t) N};
        t.raw.assign((size_t) N * 2, 0);
        const std::string p = write(dir, "f16", make("controlvector", {t}));
        run_bad("bad_type", {{p, 1.0f}}, 0, 0, 0, -1, "must be 2560 f32");
        std::remove(p.c_str());
    }
    {
        Ten t = f32dir(5, ramp_row());
        t.ne = {(uint64_t) N, 2};   // two rows: the loader takes single rows only
        t.raw.resize((size_t) N * 2 * 4, 0);
        const std::string p = write(dir, "rank2", make("controlvector", {t}));
        run_bad("bad_rows", {{p, 1.0f}}, 0, 0, 0, -1, "must be 2560 f32");
        std::remove(p.c_str());
    }
    {
        const std::string p = write(dir, "empty_range", make("controlvector", {f32dir(10, const_row(1.0f))}));
        run_bad("empty_range", {{p, 1.0f}}, 0, 20, 30, -1, "no direction in layers 20..30");
        std::remove(p.c_str());
    }
    {
        const std::string p = write(dir, "single_missing", make("controlvector", {f32dir(5, const_row(1.0f))}));
        run_bad("single_missing", {{p, 1.0f}}, 0, 0, 0, 11, "no direction for that layer");
        std::remove(p.c_str());
    }
    run_bad("missing_file", {{dir + "/fnx_cvec_load_test_absent.gguf", 1.0f}}, 0, 0, 0, -1, "cannot open");
    run_bad("no_files", {}, 0, 0, 0, -1, "no direction in layers 1..47");

    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
