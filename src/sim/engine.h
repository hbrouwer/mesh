/*
 * engine.h
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

#ifndef ENGINE_H
#define ENGINE_H

#include "network.h"

void train_network(struct network *n);
void train_network_bp(struct network *n);
void train_network_bptt(struct network *n);

void print_training_progress(struct network *n);

void scale_learning_rate(int epoch, struct network *n);
void scale_momentum(int epoch, struct network *n);

void test_network(struct network *n);
void test_network_with_item(struct network *n, struct element *e);
void test_unfolded_network(struct network *n);

#endif /* ENGINE_H */
