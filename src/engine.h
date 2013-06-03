/*
 * engine.h
 *
 * Copyright 2012, 2013 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef ENGINE_H
#define ENGINE_H

#include "network.h"
#include "session.h"

void train_network(struct network *n);

void print_training_progress(struct network *n);
void print_training_summary(struct network *n);

void scale_learning_rate(struct network *n);
void scale_momentum(struct network *n);

void train_network_with_bp(struct network *n);
void train_network_with_bptt(struct network *n);

void test_network(struct network *n);
void test_ffn_network(struct network *n);
void test_rnn_network(struct network *n);

void test_network_with_item(struct network *n, struct element *e, bool pprint);
void test_ffn_network_with_item(struct network *n, struct element *e, bool pprint);
void test_rnn_network_with_item(struct network *n, struct element *e, bool pprint);

#endif /* ENGINE_H */
