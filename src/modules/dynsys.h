/*
 * dynsys.h
 *
 * Copyright 2012-2015 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef DYNSYS_H
#define DYNSYS_H

#include "../network.h"
#include "../set.h"
#include "../vector.h"

/**************************************************************************
 *************************************************************************/
void dynsys_test_item(struct network *n, struct group *g, struct item *item);
double dynsys_processing_time(struct network *n, struct vector *a_out0,
                struct vector *a_out1);
double dynsys_unit_act(double x, double y);

#endif /* DYNSYS_H */
