/*
 * classify.h
 *
 * Copyright 2012-2016 Harm Brouwer <me@hbrouwer.eu>
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

#include "network.h"
#include "matrix.h"

/**************************************************************************
 *************************************************************************/
void confusion_matrix(struct network *n, bool print, bool pprint,
                uint32_t scheme);
void ffn_network_cm(struct network *n, bool print, bool pprint,
                uint32_t scheme);
void rnn_network_cm(struct network *n, bool print, bool pprint,
                uint32_t scheme);

/**************************************************************************
 *************************************************************************/
void print_cm_summary(struct network *n, struct matrix *cm, bool print,
                bool pprint, uint32_t scheme);

/**************************************************************************
 *************************************************************************/
void cm_signal_handler(int32_t signal);

#endif /* CLASSIFY_H */
