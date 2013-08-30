/*
 * pprint.h
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

#ifndef PPRINT_H
#define PPRINT_H

#include <stdint.h>

#include "matrix.h"
#include "vector.h"
#include "stats.h"

#define SCHEME_BLUE_RED    0
#define SCHEME_BLUE_YELLOW 1
#define SCHEME_GRAYSCALE   2
#define SCHEME_SPACEPIGS   3
#define SCHEME_MOODY_BLUES 4
#define SCHEME_FOR_JOHN    5
#define SCHEME_GRAY_ORANGE 6

/**************************************************************************
 *************************************************************************/
void pprint_vector(struct vector *v, uint32_t scheme);
void pprint_matrix(struct matrix *m, uint32_t scheme);

/**************************************************************************
 *************************************************************************/
double pprint_scale_value(double v, double min, double max);
void pprint_value_as_color(double v, uint32_t scheme);

#endif /* PPRINT_H */
