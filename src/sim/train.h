/*
 * train.h
 *
 * Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef TRAIN_H
#define TRAIN_H

#include "network.h"
#include "vector.h"

void train_network(struct network *n);
void test_network(struct network *n);
void test_unfolded_network(struct network *n);

double mean_squared_error(struct network *n);
void report_error(int epoch, double mse, struct network *n);

void feed_forward(struct network *n, struct group *g);
double unit_activation(struct network *n, struct group *g, int u);

void train_bp(struct network *n);
void train_bptt_epochwise(struct network *n);
void train_bptt_truncated(struct network *n);

struct vector *ss_output_error(struct network *n);
struct vector *ce_output_error(struct network *n);

void scale_learning_rate(int epoch, struct network *n);
void scale_momentum(int epoch, struct network *n);

void backpropagate_error(struct network *n, struct group *g,
                struct vector *error);

void comp_proj_deltas_and_error(struct network *n, struct projection *p, 
                struct vector *error);

struct vector *group_error(struct network *n, struct group *g);

void adjust_weights(struct network *n, struct group *g);
void adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p);

#endif /* TRAIN_H */
