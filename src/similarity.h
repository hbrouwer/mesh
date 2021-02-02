/*
 * Copyright 2012-2021 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef SIMILARITY_H
#define SIMILARITY_H

#include <stdint.h>

#include "network.h"
#include "matrix.h"
#include "pprint.h"

struct matrix *similarity_matrix(struct network *n);
void print_sm_summary(struct network *n, bool print_sm, bool pprint,
        enum color_scheme scheme);
void sm_signal_handler(int32_t signal);

#endif /* SIMILARITY_H */
