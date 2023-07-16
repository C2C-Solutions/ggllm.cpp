/**
 * @file falcon_main.cpp
 * @brief Falcon main application
 * https://github.com/cmp-nct/ggllm.cpp
 * MIT licensed, contributions welcome
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "falcon_common.h"
#include "libfalcon.h"
#include "build-info.h"
#include "httplib.h"
#include "json.hpp"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <signal.h>
#include <unistd.h>

static bool is_interacting = false;

using json = nlohmann::json;

falcon_context * ctx = nullptr;
falcon_model * main_model = nullptr;
gpt_params params;

std::vector<falcon_token> embd_inp; // tokenized prompt
std::vector<falcon_token> inp_system = {}; // system prompt
std::vector<falcon_token> inp_system_baseline = {}; // system differential prompt
std::vector<falcon_token> inp_pfx = {}; // prefix to user prompt
std::vector<falcon_token> inp_sfx = {}; // suffix to user prompt
std::vector<falcon_token> session_tokens;
std::vector<falcon_token> last_n_tokens;
std::vector<falcon_token> all_generation;

int n_ctx;
int n_past             = 0; // n_past tells eval() which position in KV we are at
int n_remain;
int n_consumed         = 0;
int n_session_consumed = 0;

falcon_context * prev_prev_ctx = nullptr;
falcon_context * prev_ctx = nullptr;
std::string prev_prev = "";
std::string prev = "";

bool is_antiprompt        = false;
bool input_echo           = true;

falcon_evaluation_config cfg;

void generate(const httplib::Request &req, httplib::Response &res) {
    auto body = req.body;
    auto content = json::parse(body);

    int MAX_GENERATION_LENGTH = content["max_new_tokens"];
    float temp = content["temperature"];
    float top_p = content["top_p"];
    float alpha_presence = 0.5;
    float alpha_frequency = 0.5;
    int repeat_last_n = 0;
    float tfs_z = content["tfs"];
    float top_k = content["top_k"];
    float typical_p = content["typical_p"];
    float repeat_penalty = content["repetition_penalty"];
    bool add_bos = !content["ban_eos_token"];
    auto stopping = content["stopping_strings"];
    int mirostat = 0;
    float mirostat_tau = 5;
    float mirostat_eta = 0.1;
    std::string result_string;

    std::string prompt = content["prompt"];

    if (prompt.substr(0, prev.size()) == prev) {
        prompt = prompt.substr(prev.size(), prompt.size() - prev.size());
    }

    embd_inp = falcon_tokenize(ctx, prompt, false);

    int prompt_len = embd_inp.size();

    cfg.n_tokens = prompt_len;

    falcon_eval(ctx, embd_inp.data(), cfg);

    cfg.n_past += prompt_len;

    size_t prompt_size = embd_inp.size();

    int n_remain = MAX_GENERATION_LENGTH;

    auto n_vocab = falcon_n_vocab(ctx);
    while (n_remain > 0) {
        --n_remain;

        falcon_token embd;

        // params

        falcon_token id = 0;

        {
            auto logits  = falcon_get_logits(ctx);

            // Apply params.logit_bias map
            for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                logits[it->first] += it->second;
            }

            std::vector<falcon_token_data> candidates;
            candidates.reserve(n_vocab);
            for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(falcon_token_data{token_id, logits[token_id], 0.0f});
            }

            falcon_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // Apply penalties
            float nl_logit = logits[falcon_token_nl()];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);

            llama_sample_repetition_penalty(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);

            llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);

            logits[falcon_token_nl()] = nl_logit;

            if (temp <= 0) {
                // Greedy sampling
                id = llama_sample_token_greedy(ctx, &candidates_p);
            } else {
                // Default sampling chain with temperature
                llama_sample_top_k(ctx, &candidates_p, top_k, 1);       // limit to best k (default 40)
                llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);   // remove low probability tail (default 1.0 off)
                llama_sample_typical(ctx, &candidates_p, typical_p, 1); // focus on similarities (default 1.0 off)
                llama_sample_top_p(ctx, &candidates_p, top_p, 1);       // limit to cumulative probability (default 0.95,  1.0 is off)
                llama_sample_temperature(ctx, &candidates_p, temp);     // make softmax peaky (1.0 is off, default is 0.8)
                id = llama_sample_token(ctx, &candidates_p);            // choose the token
            }
            // printf("`%d`", candidates_p.size);

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
        }

        embd = id;

        all_generation.push_back(embd);

        if (params.instruct && id == falcon_token_eos())
        {
            id = falcon_token_nl();
        }

        std::cout << falcon_token_to_str(ctx, id);

        result_string += falcon_token_to_str(ctx, id);

        bool stopped = false;

        for (auto &stopper: stopping) {
            auto pos = result_string.find(stopper);
            if (pos != std::string::npos) {
                stopped = true;
                result_string.replace(pos, 0, "");
            }
        }

        if (stopped) {
            break;
        }

        fflush(stdout);

        cfg.n_tokens = 1;

        if (falcon_eval(ctx, &embd, cfg)) {
            return;
        }

        ++cfg.n_past;

        std::string prev = content["prompt"];
    }
    json res_json;
    res_json["results"] = json::array({});
    res_json["results"].push_back(json::object());
    res_json["results"][0]["text"] = result_string;
    res.set_content(res_json.dump(), "application/json");
}

int main(int argc, char ** argv) {
    httplib::Server svr;

    cfg.n_past = 0;
    cfg.n_threads = 8;

    falcon_init_backend();

    params.n_ctx = 2048;

    ctx = falcon_init_from_gpt_params(params);
    main_model = falcon_get_falcon_model(ctx);

    params.finetune_type = falcon_detect_finetune(ctx, params.model);

    n_ctx = falcon_n_ctx(ctx);

    last_n_tokens.resize(n_ctx);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    svr.Get("/api/v1/model", [](const httplib::Request &req, httplib::Response &res) {
        res.set_content("{\"result\": \"falcon\"}", "text/plain");
    });


    svr.Post("/api/v1/generate", generate);

    svr.Get("/api/v1/stream", [](const httplib::Request &req, httplib::Response &res) {
        //res.set_content();
    });

    svr.listen("0.0.0.0", 5001);

    llama_free(ctx);

    return 0;
}