/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef TEP_H
#define TEP_H

#include "../network.h"
#include "../set.h"

double tep_iterate(struct network *n, struct group *eg, double h, double th,
        struct vector *cs, struct vector *ns,
        /* - - for recording - - */
        struct group *rg, uint32_t item_num, struct item *item,
        uint32_t event_num, FILE *fd);

void tep_test_network_with_item(struct network *n, struct group *eg, double h,
        double th, struct item *item, bool pprint,
        enum color_scheme scheme);

void tep_record_units(struct network *n, struct group *eg, double h,
        double th, struct group *rg, char *filename);

void tep_write_micro_ticks(struct network *n, struct group *eg, double h,
        double th, char *filename);

struct vector *tep_micro_ticks_for_item(struct network *n, struct group *eg,
        double h, double th, struct item *item);

void tep_signal_handler(int32_t signal);

#endif /* TEP_H */
