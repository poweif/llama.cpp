// HTTP server for DiffusionGemma: OpenAI-compatible /v1/chat/completions endpoint.
//
// Accepts the same model/inference flags as llama-diffusion-cli, plus --host and --port.
// Serialises inference (one request at a time); concurrent requests receive HTTP 503.
//
// Streaming (stream:true): each committed block is flushed as an SSE delta chunk.
// Non-streaming: the full assembled response is returned as a single JSON object.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Blocking SPSC queue used to hand SSE chunks from the inference thread to
// the httplib chunked-content provider.
// ---------------------------------------------------------------------------
struct chunk_queue {
    std::mutex              mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool                    finished = false;

    void push(std::string s) {
        { std::lock_guard<std::mutex> lk(mu); q.push_back(std::move(s)); }
        cv.notify_one();
    }

    void finish() {
        { std::lock_guard<std::mutex> lk(mu); finished = true; }
        cv.notify_one();
    }

    // Returns false only when finished and the queue is empty.
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || finished; });
        if (!q.empty()) {
            out = std::move(q.front());
            q.pop_front();
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Format one SSE data line for a chat-completion chunk delta.
// ---------------------------------------------------------------------------
static std::string make_sse_chunk(const std::string & id, const std::string & model_name,
                                  const std::string & delta, bool is_first) {
    json choice = {
        {"index",         0},
        {"delta",         is_first
                              ? json{{"role", "assistant"}, {"content", delta}}
                              : json{{"content", delta}}},
        {"finish_reason", nullptr},
    };
    json obj = {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"model",   model_name},
        {"choices", json::array({choice})},
    };
    return "data: " + obj.dump() + "\n\n";
}

static std::string make_sse_done() {
    return "data: [DONE]\n\n";
}

// ---------------------------------------------------------------------------
// Trim a denoised canvas: cut at the first eog token or the onset of a
// repetition loop (same logic as the CLI and visual server).
// ---------------------------------------------------------------------------
static size_t trim_canvas(const llama_vocab * vocab, const llama_token * canvas, size_t n) {
    size_t cut = n;
    for (size_t i = 0; i < n; i++) {
        if (llama_vocab_is_eog(vocab, canvas[i])) { cut = i; break; }
    }
    for (size_t i = 0; i + 1 < cut; i++) {
        for (size_t stride = 1; stride <= 2; stride++) {
            size_t reps = 0;
            for (size_t j = i; j + stride < cut && canvas[j] == canvas[j + stride]; j += stride) { reps++; }
            if (reps >= 6) { cut = i; goto done; }
        }
    }
done:
    return cut;
}

// ---------------------------------------------------------------------------
// Inference state shared across requests (loaded once at startup).
// ---------------------------------------------------------------------------
struct server_state {
    llama_model *             model        = nullptr;
    llama_context *           ctx          = nullptr;
    const llama_vocab *       vocab        = nullptr;
    common_chat_templates_ptr chat_templates;
    diffusion_eb_params       base_eb;
    int64_t                   canvas_length = 0;
    int                       maxtok        = 0;
    std::string               model_name;
    std::atomic<bool>         inferring{false};
};

// ---------------------------------------------------------------------------
// Run one inference turn and push completed blocks through `out`.
// Called from a detached thread for streaming, or inline for non-streaming.
// Returns the full assembled response text.
// ---------------------------------------------------------------------------
struct gen_request {
    std::vector<common_chat_msg> messages;
    int                          seed     = 0;
    int                          n_blocks = 1;
};

static std::string run_generation(server_state & st,
                                  const gen_request & req,
                                  chunk_queue * stream_q,   // null for non-streaming
                                  const std::string & req_id) {
    // Apply chat template and tokenize.
    common_chat_templates_inputs inputs;
    inputs.messages              = req.messages;
    inputs.add_generation_prompt = true;
    const std::string prompt     = common_chat_templates_apply(st.chat_templates.get(), inputs).prompt;
    std::vector<llama_token> prefix = common_tokenize(st.vocab, prompt, true, true);

    std::vector<llama_token> output_tokens((size_t) st.maxtok);
    std::vector<llama_token> answer;
    std::string              full_text;
    bool                     first_block = true;

    for (int b = 0; b < std::max(1, req.n_blocks); b++) {
        const int32_t prefix_len = (int32_t) prefix.size();
        const int32_t max_length = prefix_len + (int32_t) st.canvas_length;
        if (max_length > st.maxtok) { break; }

        diffusion_eb_params eb   = st.base_eb;
        eb.max_length            = max_length;
        eb.seed                  = req.seed + b;
        eb.visual_mode           = false;
        eb.step_callback         = nullptr;
        eb.step_callback_user_data = nullptr;

        int32_t n_generated = 0;
        diffusion_generate_entropy_bound(st.ctx, prefix.data(), output_tokens.data(),
                                         prefix_len, eb, n_generated);
        if (n_generated <= prefix_len) { break; }

        const llama_token * canvas = output_tokens.data() + prefix_len;
        const size_t        cut    = trim_canvas(st.vocab, canvas, (size_t) st.canvas_length);

        answer.insert(answer.end(), canvas, canvas + cut);
        // special=true to preserve <|channel|> reasoning markers for clients that want them
        const std::string block_text = common_detokenize(st.vocab, answer, true);

        if (stream_q) {
            // delta = new text appended by this block
            const std::string delta = block_text.substr(full_text.size());
            stream_q->push(make_sse_chunk(req_id, st.model_name, delta, first_block));
        }
        full_text   = block_text;
        first_block = false;

        if (cut < (size_t) st.canvas_length) { break; }
        prefix.insert(prefix.end(), canvas, canvas + cut);
    }

    if (stream_q) {
        stream_q->push(make_sse_done());
        stream_q->finish();
    }
    return full_text;
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

static void handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

static void handle_models(const httplib::Request &, httplib::Response & res,
                          const std::string & model_name) {
    json body = {
        {"object", "list"},
        {"data",   json::array({
            json{{"id", model_name}, {"object", "model"}, {"owned_by", "user"}},
        })},
    };
    res.set_content(body.dump(), "application/json");
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res,
                                    server_state & st) {
    // Reject concurrent requests: diffusion context is not thread-safe.
    bool expected = false;
    if (!st.inferring.compare_exchange_strong(expected, true)) {
        res.status = 503;
        res.set_content(R"({"error":{"message":"Server busy","type":"server_busy"}})",
                        "application/json");
        return;
    }

    // Ensure the busy flag is cleared when we exit, regardless of path.
    struct guard {
        std::atomic<bool> & flag;
        ~guard() { flag.store(false); }
    } busy_guard{st.inferring};

    // Parse request body.
    gen_request gr;
    bool        stream = false;
    std::string req_id;
    try {
        const json body = json::parse(req.body);
        stream    = body.value("stream", false);
        gr.seed   = body.value("seed",     0);
        gr.n_blocks = body.value("n_blocks", 1);
        req_id    = body.value("id", std::string("chatcmpl-") + std::to_string(ggml_time_us()));
        gr.messages = common_chat_msgs_parse_oaicompat(body.at("messages"));
    } catch (const std::exception & e) {
        res.status = 400;
        json err = {{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (stream) {
        // Streaming: run inference in a background thread, stream blocks as SSE.
        auto q = std::make_shared<chunk_queue>();

        // The content provider is called repeatedly by httplib's send thread; it blocks on q->pop().
        // We capture st by reference — safe because st outlives all requests.
        res.set_chunked_content_provider(
            "text/event-stream",
            [q](size_t, httplib::DataSink & sink) -> bool {
                std::string chunk;
                if (q->pop(chunk)) {
                    return sink.write(chunk.data(), chunk.size());
                }
                sink.done();
                return false;
            }
        );

        // Launch inference; the lambda owns q via shared_ptr, st is global.
        std::thread([&st, gr, q, req_id]() mutable {
            run_generation(st, gr, q.get(), req_id);
        }).detach();

    } else {
        // Non-streaming: run inference synchronously, return full JSON.
        const std::string text = run_generation(st, gr, nullptr, req_id);

        json choice = {
            {"index",         0},
            {"message",       {{"role", "assistant"}, {"content", text}}},
            {"finish_reason", "stop"},
        };
        json body = {
            {"id",      req_id},
            {"object",  "chat.completion"},
            {"model",   st.model_name},
            {"choices", json::array({choice})},
        };
        res.set_content(body.dump(), "application/json");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    // Pre-extract --host and --port before common_params_parse, which does not
    // expose those flags under LLAMA_EXAMPLE_DIFFUSION.
    std::string     host = "127.0.0.1";
    int             port = 8080;
    std::vector<char *> fwd;
    for (int i = 0; i < argc; i++) {
        if ((strcmp(argv[i], "--host") == 0 || strcmp(argv[i], "-H") == 0) && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else {
            fwd.push_back(argv[i]);
        }
    }
    int fwd_argc = (int) fwd.size();

    common_params params;
    common_init();
    if (!common_params_parse(fwd_argc, fwd.data(), params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    ggml_time_init();
    llama_backend_init();

    llama_model_params mparams   = llama_model_default_params();
    mparams.n_gpu_layers         = params.n_gpu_layers;
    mparams.devices              = params.devices.data();
    mparams.use_mmap             = params.use_mmap;
    mparams.use_mlock            = params.use_mlock;
    if (!params.tensor_buft_overrides.empty()) {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr);
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("model is not a diffusion model\n");
        llama_model_free(model);
        return 1;
    }

    char canvas_str[32];
    int64_t canvas_length = 0;
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
        canvas_length = strtol(canvas_str, nullptr, 10);
    }
    if (canvas_length <= 0) {
        LOG_ERR("model has no diffusion.canvas_length — only canvas block-diffusion models are supported\n");
        llama_model_free(model);
        return 1;
    }

    // Enable self-conditioning graph before context creation.
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);

    // Device enumeration: count GPU devices from the user's --device selection when provided, falling
    // back to all enumerated devices otherwise. Counting selected devices avoids treating one physical
    // GPU exposed through multiple backends (e.g. ROCm + Vulkan on the same iGPU) as multiple GPUs.
    int                gpu_devs = 0;
    ggml_backend_dev_t gpu_dev  = nullptr;
    const bool have_selected = !params.devices.empty() && params.devices[0] != nullptr;
    if (have_selected) {
        for (auto * d : params.devices) {
            if (!d) break;
            const auto dt = ggml_backend_dev_type(d);
            if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_devs++;
                if (!gpu_dev) gpu_dev = d;
            }
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto dt = ggml_backend_dev_type(d);
            if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_devs++;
                if (!gpu_dev) gpu_dev = d;
            }
        }
    }
    const bool one_gpu = (gpu_devs <= 1);

    const int n_head = std::max(1, (int) llama_model_n_head(model));
    auto make_cparams = [&](int n) {
        llama_context_params c = llama_context_default_params();
        c.n_ctx         = (uint32_t) n;
        c.n_batch       = (uint32_t) n;
        const int chunk = (int) std::clamp<int64_t>((int64_t(1) << 30) / ((int64_t) n_head * n), 256, 2048);
        c.n_ubatch      = (uint32_t) std::min(n, chunk);
        c.n_outputs_max = (uint32_t) canvas_length;
        c.no_perf       = true;
        c.flash_attn_type = params.flash_attn_type;
        return c;
    };

    // Auto-size context: probe the largest canvas-aligned context that fits VRAM, then RAM.
    const int n_ctx_train = (int) llama_model_n_ctx_train(model);
    const int floor_ctx   = std::max((int) canvas_length * 4, 2048);
    const int auto_ceil   = n_ctx_train > 0 ? std::min(n_ctx_train, 65536) : 65536;
    const int cands[]     = {65536, 49152, 40960, 32768, 24576, 20480, 16384, 12288, 8192, 6144, 4096, 2048};

    size_t v_free = 0, v_total = 0;
    if (gpu_dev) ggml_backend_dev_memory(gpu_dev, &v_free, &v_total);
    size_t r_free = 0, r_total = 0;
    if (ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        ggml_backend_dev_memory(cpu_dev, &r_free, &r_total);
    }
    const size_t weights    = llama_model_size(model);
    const size_t ram_budget = r_free > weights ? (size_t) ((r_free - weights) * 0.7) : 0;

    auto probe = [&](int ceil_ctx, size_t budget, size_t gpu_headroom, int * out_n) -> llama_context * {
        for (int raw : cands) {
            if (raw > ceil_ctx) { continue; }
            int N = (int) ((raw / canvas_length) * canvas_length);
            if (N < floor_ctx) { break; }
            if (budget) {
                const double min_scores = (double) n_head * (double) N * (double) N * 4.0;
                if (min_scores > (double) budget * 0.9) { continue; }
            }
            llama_context * c = llama_init_from_model(model, make_cparams(N));
            if (!c) { continue; }
            if (gpu_headroom && gpu_dev) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(gpu_dev, &f, &t);
                if (f < gpu_headroom) { llama_free(c); continue; }
            }
            *out_n = N;
            return c;
        }
        return nullptr;
    };

    // If -n/--n-predict was set, honour it as the context ceiling.
    const int n_predict_ceil = params.n_predict > 0
                                   ? (int) ((params.n_predict / canvas_length + 1) * canvas_length + 2048)
                                   : 0;
    const int ceil_ctx = n_predict_ceil > 0 ? std::min(auto_ceil, n_predict_ceil) : auto_ceil;

    int             maxtok = 0;
    llama_context * ctx    = nullptr;
    const char *    reason = "auto";

    {
        const size_t vram_headroom = v_total ? (size_t) (v_total * 0.08) : (size_t) 1536 * 1024 * 1024;
        int n1 = 0;
        ctx = probe(ceil_ctx, v_free, vram_headroom, &n1);
        if (ctx) { maxtok = n1; reason = "vram"; }
        if ((!ctx || n1 < ceil_ctx) && ram_budget > v_free) {
            int n2 = 0;
            llama_context * c = probe(ceil_ctx, ram_budget, 0, &n2);
            if (c && n2 > maxtok) { if (ctx) { llama_free(ctx); } ctx = c; maxtok = n2; reason = "ram"; }
            else if (c) { llama_free(c); }
        }
    }
    if (!ctx) {
        int N = std::max((int) canvas_length, (int) ((floor_ctx / canvas_length) * canvas_length));
        ctx = llama_init_from_model(model, make_cparams(N));
        if (ctx) { maxtok = N; reason = "floor"; }
    }
    if (!ctx) {
        LOG_ERR("failed to allocate a context (VRAM free=%zu MiB, RAM budget=%zu MiB)\n",
                v_free / (1024 * 1024), ram_budget / (1024 * 1024));
        llama_model_free(model);
        return 1;
    }
    llama_set_causal_attn(ctx, false);

    // Entropy-bound params from GGUF metadata + reference defaults.
    auto meta_f = [&](const char * key, float def) -> float {
        char buf[32];
        return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
    };
    auto meta_i = [&](const char * key, int32_t def) -> int32_t {
        char buf[32];
        return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0
                   ? (int32_t) strtol(buf, nullptr, 10) : def;
    };

    diffusion_eb_params base_eb;
    base_eb.max_denoising_steps  = meta_i("diffusion.eb_max_steps",           48);
    base_eb.t_min                = meta_f("diffusion.eb_t_min",               0.4f);
    base_eb.t_max                = meta_f("diffusion.eb_t_max",               0.8f);
    base_eb.entropy_bound        = meta_f("diffusion.eb_entropy_bound",       0.1f);
    base_eb.stability_threshold  = meta_i("diffusion.eb_stability_threshold", 1);
    base_eb.confidence_threshold = meta_f("diffusion.eb_confidence_threshold",0.005f);
    // CLI overrides (e.g. --diffusion-eb-t-min etc.) are honoured via common_params.diffusion
    if (params.diffusion.eb_max_steps     >  0) { base_eb.max_denoising_steps  = params.diffusion.eb_max_steps; }
    if (params.diffusion.eb_t_min         >= 0) { base_eb.t_min                = params.diffusion.eb_t_min; }
    if (params.diffusion.eb_t_max         >= 0) { base_eb.t_max                = params.diffusion.eb_t_max; }
    if (params.diffusion.eb_entropy_bound >= 0) { base_eb.entropy_bound        = params.diffusion.eb_entropy_bound; }
    if (params.diffusion.eb_stability     >= 0) { base_eb.stability_threshold  = params.diffusion.eb_stability; }
    if (params.diffusion.eb_confidence    >= 0) { base_eb.confidence_threshold = params.diffusion.eb_confidence; }
    base_eb.kv_cache          = one_gpu;
    base_eb.gpu_sampling      = one_gpu;
    base_eb.gpu_sample_reduce = one_gpu;

    // Derive a short model name from the filename.
    std::string model_name = params.model.path;
    {
        const size_t slash = model_name.rfind('/');
        if (slash != std::string::npos) { model_name = model_name.substr(slash + 1); }
        const size_t dot = model_name.rfind('.');
        if (dot != std::string::npos) { model_name = model_name.substr(0, dot); }
    }

    server_state st;
    st.model          = model;
    st.ctx            = ctx;
    st.vocab          = llama_model_get_vocab(model);
    st.chat_templates = common_chat_templates_init(model, "");
    st.base_eb        = base_eb;
    st.canvas_length  = canvas_length;
    st.maxtok         = maxtok;
    st.model_name     = model_name;

    LOG_INF("diffusion-server: model=%s canvas=%d maxtok=%d (%s) ngl=%d gpu_sampling=%s kv_cache=%s\n",
            model_name.c_str(), (int) canvas_length, maxtok, reason,
            params.n_gpu_layers, base_eb.gpu_sampling ? "on" : "off", base_eb.kv_cache ? "on" : "off");
    LOG_INF("diffusion-server: listening on %s:%d\n", host.c_str(), port);

    httplib::Server svr;
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        try { std::rethrow_exception(ep); } catch (const std::exception & e) {
            res.status = 500;
            json err = {{"error", {{"message", e.what()}, {"type", "internal_error"}}}};
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Get("/health",           [](const httplib::Request & req, httplib::Response & res) {
        handle_health(req, res);
    });
    svr.Get("/v1/models",        [&](const httplib::Request & req, httplib::Response & res) {
        handle_models(req, res, st.model_name);
    });
    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        handle_chat_completions(req, res, st);
    });

    if (!svr.listen(host, port)) {
        LOG_ERR("failed to listen on %s:%d\n", host.c_str(), port);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
