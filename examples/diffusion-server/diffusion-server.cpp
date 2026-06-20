// HTTP server for DiffusionGemma: OpenAI-compatible /v1/chat/completions endpoint.
//
// Accepts the same model/inference flags as llama-diffusion-cli, plus --host and --port.
// Serialises inference (one request at a time); concurrent requests receive HTTP 503.
//
// Streaming (stream:true): each committed block is flushed as an SSE delta chunk.
// Non-streaming: the full assembled response is returned as a single JSON object.
//
// Tool calls: fully supported. Tools are passed to the chat template; the generated
// response is parsed with common_chat_parse to extract any tool_calls and format them
// in the OpenAI-compatible response. When tools are present in a streaming request,
// text is buffered during generation and emitted after parsing to avoid leaking raw
// tool-call syntax in the content stream.
//
// Image input: not supported — DiffusionGemma has no vision encoder (mmproj). Any
// request containing image_url content parts is rejected with HTTP 400.

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
// SSE helpers
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

static std::string make_sse_finish_chunk(const std::string & id, const std::string & model_name,
                                         const std::string & finish_reason) {
    json choice = {
        {"index",         0},
        {"delta",         json::object()},
        {"finish_reason", finish_reason},
    };
    json obj = {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"model",   model_name},
        {"choices", json::array({choice})},
    };
    return "data: " + obj.dump() + "\n\n";
}

static std::string make_sse_tool_call_chunk(const std::string & id, const std::string & model_name,
                                            const common_chat_tool_call & tc, size_t idx) {
    json tc_json = {
        {"index",    (int) idx},
        {"id",       tc.id},
        {"type",     "function"},
        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}},
    };
    json choice = {
        {"index",         0},
        {"delta",         {{"tool_calls", json::array({tc_json})}}},
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
// One generation request: pre-formatted prompt + sampling knobs.
// The caller applies the chat template (with tools if any) before creating
// this struct so that run_generation can stay format-agnostic.
// ---------------------------------------------------------------------------
struct gen_request {
    std::string prompt;
    int         seed     = 0;
    int         n_blocks = 1;
};

// ---------------------------------------------------------------------------
// Run one inference turn and push committed blocks as SSE deltas through
// `stream_q` (null for non-streaming / buffered-tool-call mode).
// Returns the full assembled response text.
// Does NOT push [DONE] or call stream_q->finish() — the caller is responsible.
// Strip DiffusionGemma thinking channels: <|channel>NAME\nCONTENT<channel|>ACTUAL_RESPONSE
// Returns everything after the last <channel|>, trimming any leading whitespace.
// If no thinking channel is present, returns the text unchanged.
static std::string strip_thinking(const std::string & text) {
    const std::string close = "<channel|>";
    // Use rfind: the model may generate multiple channel blocks (e.g. an empty
    // thought block followed by an unlabelled content block ending with another
    // <channel|>).  The actual response always follows the LAST close tag.
    size_t pos = text.rfind(close);
    if (pos != std::string::npos) {
        std::string after = text.substr(pos + close.size());
        size_t start = after.find_first_not_of(" \t\n\r");
        return start == std::string::npos ? "" : after.substr(start);
    }
    // No close tag: canvas ran out of tokens inside a thinking block.
    // Strip the "<|channel>NAME\n" opening so the caller gets the text content
    // rather than leaking raw special-token syntax to the client.
    const std::string open = "<|channel>";
    if (text.substr(0, open.size()) == open) {
        size_t nl = text.find('\n');
        if (nl != std::string::npos) {
            std::string rest = text.substr(nl + 1);
            size_t start = rest.find_first_not_of(" \t\n\r");
            return start == std::string::npos ? "" : rest.substr(start);
        }
    }
    return text;
}

// ---------------------------------------------------------------------------
static std::string run_generation(server_state & st,
                                  const gen_request & req,
                                  chunk_queue * stream_q,
                                  const std::string & req_id) {
    std::vector<llama_token> prefix = common_tokenize(st.vocab, req.prompt, true, true);

    std::vector<llama_token> output_tokens((size_t) st.maxtok);
    std::vector<llama_token> answer;
    std::string              full_text;
    bool                     first_block   = true;
    bool                     past_thinking = false; // true once we have passed <channel|>

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
        // special=true so <|channel>/<channel|> markers are present for stripping below
        const std::string block_text = common_detokenize(st.vocab, answer, true);

        if (stream_q) {
            if (!past_thinking) {
                // Look for the thinking-channel close tag in the cumulative text.
                const std::string close = "<channel|>";
                // rfind: same reason as strip_thinking — actual response is after the LAST close.
                size_t close_pos = block_text.rfind(close);
                if (close_pos != std::string::npos) {
                    past_thinking = true;
                    // Emit only the visible portion after the close tag (trim leading whitespace).
                    std::string visible = block_text.substr(close_pos + close.size());
                    size_t vstart = visible.find_first_not_of(" \t\n\r");
                    if (vstart != std::string::npos) {
                        stream_q->push(make_sse_chunk(req_id, st.model_name,
                                                      visible.substr(vstart), first_block));
                        first_block = false;
                    }
                }
                // else: still inside the thinking block — suppress output until close tag appears
            } else {
                // Already past thinking — stream new delta normally.
                const std::string delta = block_text.substr(full_text.size());
                if (!delta.empty()) {
                    stream_q->push(make_sse_chunk(req_id, st.model_name, delta, first_block));
                    first_block = false;
                }
            }
        }
        full_text = block_text;

        if (cut < (size_t) st.canvas_length) { break; }
        prefix.insert(prefix.end(), canvas, canvas + cut);
    }

    // Always return the clean (thinking-stripped) text so the caller can parse/forward it.
    return strip_thinking(full_text);
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

// Scan a messages array for image_url / image content parts.
// Returns a non-empty error string if any are found (with minimal format validation),
// or empty string if clean.
static std::string check_no_image_parts(const json & messages) {
    if (!messages.is_array()) { return {}; }
    for (const auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("content")) { continue; }
        const auto & content = msg.at("content");
        if (!content.is_array()) { continue; }
        for (const auto & part : content) {
            if (!part.is_object() || !part.contains("type")) { continue; }
            const auto & type = part.at("type");
            if (type == "image_url" || type == "image") {
                // Validate image_url structure before rejecting so callers get a useful error.
                if (type == "image_url") {
                    if (!part.contains("image_url") || !part.at("image_url").is_object()) {
                        return "Malformed image_url content part: missing or non-object 'image_url' field";
                    }
                    if (!part.at("image_url").contains("url")) {
                        return "Malformed image_url content part: missing 'url' field";
                    }
                }
                return "Image input is not supported: this model has no vision encoder. "
                       "Remove image content parts from the request.";
            }
        }
    }
    return {};
}

// Parse the generated text to extract tool calls and clean content.
// Loads the serialised PEG arena from cp.parser so the model-specific format
// (Gemma 4 thinking channel, tool call delimiters, generation-prompt prefix) is
// handled correctly.  Falls back to treating full_text as plain content on failure.
static common_chat_msg parse_response(const std::string & full_text, const common_chat_params & cp) {
    common_chat_msg parsed;
    try {
        common_chat_parser_params pp(cp);
        if (!cp.parser.empty()) {
            pp.parser.load(cp.parser);
        }
        parsed = common_chat_parse(full_text, /*is_partial=*/false, pp);
    } catch (...) {
        parsed.content = full_text;
    }
    // Assign stable IDs to any tool calls that didn't emit one.
    std::vector<std::string> ids;
    parsed.set_tool_call_ids(ids, [&] {
        return std::string("call_") + std::to_string(ggml_time_us());
    });
    return parsed;
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res,
                                    server_state & st) {
    bool expected = false;
    if (!st.inferring.compare_exchange_strong(expected, true)) {
        res.status = 503;
        res.set_content(R"({"error":{"message":"Server busy","type":"server_busy"}})",
                        "application/json");
        return;
    }
    // busy_guard resets st.inferring on scope exit. For streaming requests we
    // transfer ownership into the generation thread so the flag stays true until
    // inference actually finishes (not just until the HTTP handler returns).
    bool thread_owns_guard = false;
    struct guard { std::atomic<bool> & flag; bool & skip; ~guard() { if (!skip) flag.store(false); } }
        busy_guard{st.inferring, thread_owns_guard};

    gen_request      gr;
    bool             stream      = false;
    bool             has_tools   = false;
    std::string      req_id;
    common_chat_params chat_params;

    try {
        const json body = json::parse(req.body);
        stream      = body.value("stream", false);
        gr.seed     = body.value("seed",     0);
        // Target ~256 tokens of generation budget regardless of canvas_length so the
        // model has room to think in early blocks and still produce a visible response.
        // Callers can override via "n_blocks" in the request body.
        const int default_n_blocks = std::max(1, 256 / (int) st.canvas_length);
        gr.n_blocks = body.value("n_blocks", default_n_blocks);
        req_id      = body.value("id", std::string("chatcmpl-") + std::to_string(ggml_time_us()));

        const json & msg_arr = body.at("messages");

        // Reject image content — no vision encoder is available.
        const std::string img_err = check_no_image_parts(msg_arr);
        if (!img_err.empty()) {
            res.status = 400;
            json err = {{"error", {{"message", img_err}, {"type", "invalid_request_error"}}}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto messages = common_chat_msgs_parse_oaicompat(msg_arr);

        // Build template inputs, injecting tools if provided.
        common_chat_templates_inputs tmpl_inputs;
        tmpl_inputs.messages              = std::move(messages);
        tmpl_inputs.add_generation_prompt = true;

        if (body.contains("tools") && body.at("tools").is_array() && !body.at("tools").empty()) {
            tmpl_inputs.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
            has_tools = true;
        }
        if (body.contains("tool_choice") && body.at("tool_choice").is_string()) {
            tmpl_inputs.tool_choice =
                common_chat_tool_choice_parse_oaicompat(body.at("tool_choice"));
        }

        chat_params = common_chat_templates_apply(st.chat_templates.get(), tmpl_inputs);
        gr.prompt   = chat_params.prompt;

    } catch (const std::exception & e) {
        res.status = 400;
        json err = {{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (stream) {
        auto q = std::make_shared<chunk_queue>();

        res.set_chunked_content_provider(
            "text/event-stream",
            [q](size_t, httplib::DataSink & sink) -> bool {
                std::string chunk;
                if (q->pop(chunk)) { return sink.write(chunk.data(), chunk.size()); }
                sink.done();
                return false;
            }
        );

        // Transfer the busy flag to the generation thread: the flag must stay true
        // until inference finishes, not just until the HTTP handler returns.
        thread_owns_guard = true;
        // When tools are present we suppress per-block streaming so raw tool-call syntax
        // never reaches the client as content; we emit clean deltas after parsing.
        std::thread([&st, gr, q, req_id, chat_params, has_tools]() mutable {
            // Reset st.inferring when this thread exits (transferred from handler guard).
            struct thread_guard { std::atomic<bool> & f; ~thread_guard() { f.store(false); } }
                tg{st.inferring};
            try {
                chunk_queue * gen_q  = has_tools ? nullptr : q.get();
                std::string full_text = run_generation(st, gr, gen_q, req_id);

                if (has_tools) {
                    const auto parsed = parse_response(full_text, chat_params);

                    // Prefer parsed.content; fall back to the stripped full_text when
                    // the PEG parser found no content (e.g. plain response with no tool call).
                    const std::string & content =
                        (parsed.content.empty() && parsed.tool_calls.empty())
                            ? full_text : parsed.content;

                    if (!content.empty()) {
                        q->push(make_sse_chunk(req_id, st.model_name, content, true));
                    }

                    // Emit one chunk per tool call.
                    for (size_t i = 0; i < parsed.tool_calls.size(); i++) {
                        q->push(make_sse_tool_call_chunk(req_id, st.model_name,
                                                         parsed.tool_calls[i], i));
                    }

                    const std::string reason =
                        parsed.tool_calls.empty() ? "stop" : "tool_calls";
                    q->push(make_sse_finish_chunk(req_id, st.model_name, reason));
                } else {
                    q->push(make_sse_finish_chunk(req_id, st.model_name, "stop"));
                }

                q->push(make_sse_done());
            } catch (...) {
                q->push("data: {\"error\":{\"message\":\"Internal generation error\","
                        "\"type\":\"internal_error\"}}\n\n");
                q->push(make_sse_done());
            }
            q->finish();
        }).detach();

    } else {
        const std::string full_text = run_generation(st, gr, nullptr, req_id);

        // Only invoke the PEG parser when tools are present. Without tools,
        // return full_text directly — the parser would otherwise prepend the
        // generation_prompt into the content field.
        json        msg_json      = {{"role", "assistant"}};
        std::string finish_reason = "stop";

        if (has_tools) {
            const auto parsed = parse_response(full_text, chat_params);
            const std::string & text = parsed.content.empty() ? full_text : parsed.content;
            if (parsed.tool_calls.empty()) {
                msg_json["content"] = text;
            } else {
                finish_reason          = "tool_calls";
                msg_json["content"]    = text.empty() ? json(nullptr) : json(text);
                json tc_arr = json::array();
                for (const auto & tc : parsed.tool_calls) {
                    tc_arr.push_back({
                        {"id",       tc.id},
                        {"type",     "function"},
                        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}},
                    });
                }
                msg_json["tool_calls"] = tc_arr;
            }
        } else {
            msg_json["content"] = full_text;
        }

        json choice = {
            {"index",         0},
            {"message",       msg_json},
            {"finish_reason", finish_reason},
        };
        json resp_body = {
            {"id",      req_id},
            {"object",  "chat.completion"},
            {"model",   st.model_name},
            {"choices", json::array({choice})},
        };
        res.set_content(resp_body.dump(), "application/json");
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
    mparams.kv_overrides         = params.kv_overrides.empty() ? nullptr : params.kv_overrides.data();
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
    for (const auto & ov : params.kv_overrides) {
        if (strcmp(ov.key, "diffusion.canvas_length") == 0 && ov.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
            canvas_length = ov.val_i64;
            break;
        }
    }
    if (canvas_length <= 0 && llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
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

    // Pre-allocate the PKV store to full capacity. Without this, the first PREFILL request allocates a
    // small store; a later request with a longer prompt frees and recreates it inside build_graph, which
    // corrupts the galloc hash (old pointers remain; new tensors look "new") and crashes the Vulkan backend.
    // Note: the first request will log a few ggml_gallocr_needs_realloc errors for pkv_k/leaf tensors;
    // these are benign — ggml automatically re-reserves and retries — and only appear on the first call.
    if (base_eb.kv_cache) {
        llama_diffusion_preallocate_pkv_store(ctx, maxtok);
    }

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
