/*
 * erps.h
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

#ifndef ERPS_H
#define ERPS_H

#include "network.h"
#include "vector.h"

void compute_erp_correlates(struct network *n);

double compute_n400_correlate(struct vector *v, struct vector *pv);
double compute_p600_correlate(struct vector *v, struct vector *pv); 

#endif /* ERPS_H */