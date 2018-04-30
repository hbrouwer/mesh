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

#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <stdint.h>

#include "matrix.h"
#include "network.h"
#include "pprint.h"

struct matrix *confusion_matrix(struct network *n);
struct matrix *ffn_network_cm(struct network *n);
struct matrix *rnn_network_cm(struct network *n);

void classify(struct vector *ov, struct vector *tv, struct matrix *cm);

void print_cm_summary(struct network *n, bool print_cm, bool pprint,
        enum color_scheme scheme);

void cm_signal_handler(int32_t signal);

#endif /* CLASSIFY_H */
