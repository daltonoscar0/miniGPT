#pragma once

// ============================================================
//  AdamW.hpp, AdamW optimizer
//
//  AdamW = Adam with decoupled weight decay.
//  (Loshchilov & Hutter, 2019)
//
//  For each parameter θ and its gradient g:
//
//    m_t = β1·m_{t-1} + (1-β1)·g          (first moment)
//    v_t = β2·v_{t-1} + (1-β2)·g²          (second moment)
//    m̂_t = m_t / (1 - β1^t)               (bias correction)
//    v̂_t = v_t / (1 - β2^t)
//    θ_t = θ_{t-1}·(1 - lr·λ) - lr·m̂_t/√(v̂_t+ε)
//
//  The key difference from Adam: weight decay (λ) is applied
//  directly to the parameter, not to the gradient.  This means
//  weight decay acts as a true L2 regulariser independent of
//  the adaptive learning rate scaling.
//
//  Defaults
//  ────────
//  lr    = 3e-4   (good default for transformers)
//  β1    = 0.9
//  β2    = 0.999
//  ε     = 1e-8
//  λ     = 0.01   (weight decay)
//
//  Usage
//  ─────
//  AdamW opt(lr);
//  opt.register_param(W, dW);   // register all param/grad pairs
//  ...
//  // training loop:
//  model.zero_grad();
//  loss = model.forward_loss(...);
//  model.backward(...);
//  model.clip_grad_norm(1.0f);
//  opt.step();                  // updates all registered params
// ============================================================

#include "../embed/Embedding.hpp"

#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace miniGPT {

// ── ParamGroup ────────────────────────────────────────────
struct ParamGroup {
    Matrix2D* param;
    Matrix2D* grad;
    Matrix2D  m;
    Matrix2D  v;

    ParamGroup(Matrix2D* p, Matrix2D* g)
        : param(p), grad(g)
        , m(p->rows, p->cols, 0.f)
        , v(p->rows, p->cols, 0.f)
    {}
};

// ── AdamW ─────────────────────────────────────────────────
class AdamW {
public:
    explicit AdamW(float lr    = 3e-4f,
                   float beta1 = 0.9f,
                   float beta2 = 0.999f,
                   float eps   = 1e-8f,
                   float wd    = 0.01f)
        : lr_(lr), beta1_(beta1), beta2_(beta2)
        , eps_(eps), wd_(wd), t_(0)
    {
        if (lr <= 0.f)
            throw std::invalid_argument("AdamW: lr must be > 0");
    }

    void register_param(Matrix2D& param, Matrix2D& grad,
                        bool decay = true)
    {
        groups_.emplace_back(&param, &grad);
        apply_decay_.push_back(decay);
    }

    void step() {
        ++t_;
        float bc1 = 1.f - std::pow(beta1_, static_cast<float>(t_));
        float bc2 = 1.f - std::pow(beta2_, static_cast<float>(t_));

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            auto& pg    = groups_[i];
            bool  decay = apply_decay_[i];

            Matrix2D& W  = *pg.param;
            Matrix2D& dW = *pg.grad;
            Matrix2D& m  = pg.m;
            Matrix2D& v  = pg.v;

            for (std::size_t j = 0; j < W.data.size(); ++j) {
                float g = dW.data[j];
                m.data[j] = beta1_ * m.data[j] + (1.f - beta1_) * g;
                v.data[j] = beta2_ * v.data[j] + (1.f - beta2_) * g * g;
                float m_hat   = m.data[j] / bc1;
                float v_hat   = v.data[j] / bc2;
                float wd_term = decay ? wd_ * W.data[j] : 0.f;
                W.data[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_)
                                    + wd_term);
            }
        }
    }

    float lr()         const noexcept { return lr_; }
    int   step_count() const noexcept { return t_; }
    void  set_lr(float lr)            { lr_ = lr; }

    std::size_t n_params() const {
        std::size_t n = 0;
        for (const auto& pg : groups_) n += pg.param->data.size();
        return n;
    }

private:
    float lr_, beta1_, beta2_, eps_, wd_;
    int   t_;
    std::vector<ParamGroup> groups_;
    std::vector<bool>       apply_decay_;
};

} // namespace miniGPT
