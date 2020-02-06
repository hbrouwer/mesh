/*
 * Copyright 2012-2020 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef ERP_H
#define ERP_H

#include "../network.h"
#include "../set.h"

void erp_contrast(struct network *n, struct group *gen,
        struct item *ctl, struct item *tgt);
void erp_write_values(struct network *n, struct group *N400_gen,
        struct group *P600_gen, char *filename);
struct vector *erp_values_for_item(struct network *n, struct group *g,
        struct item *item);

#endif /* ERP_H */
