/*
 * Copyright 2012-2018 Harm Brouwer <me@hbrouwer.eu>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ACT_H
#define ACT_H

#include <stdint.h>

#include "math.h"
#include "network.h"

/* use fast_exp() or exp() */
#ifdef FAST_EXP
#define EXP(x) fast_exp(x)
#else
#define EXP(x) exp(x)
#endif /* FAST_EXP */

void feed_forward(struct network *n, struct group *g);

double act_fun_logistic(struct group *g, uint32_t i);
double act_fun_logistic_deriv(struct group *g, uint32_t i);

double act_fun_bipolar_sigmoid(struct group *g, uint32_t i);
double act_fun_bipolar_sigmoid_deriv(struct group *g, uint32_t i);

double act_fun_softmax(struct group *g, uint32_t i);
double act_fun_softmax_deriv(struct group *g, uint32_t i);

double act_fun_tanh(struct group *g, uint32_t i);
double act_fun_tanh_deriv(struct group *g, uint32_t i);

double act_fun_linear(struct group *g, uint32_t i);
double act_fun_linear_deriv(struct group *g, uint32_t i);

double act_fun_softplus(struct group *g, uint32_t i);
double act_fun_softplus_deriv(struct group *g, uint32_t i);

double act_fun_relu(struct group *g, uint32_t i);
double act_fun_relu_deriv(struct group *g, uint32_t i);

double act_fun_leaky_relu(struct group *g, uint32_t i);
double act_fun_leaky_relu_deriv(struct group *g, uint32_t i);

double act_fun_elu(struct group *g, uint32_t i);
double act_fun_elu_deriv(struct group *g, uint32_t i);

#endif /* ACT_H */
