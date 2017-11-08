/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

/* color scheme */
enum color_scheme
{
        scheme_blue_red,
        scheme_blue_yellow,
        scheme_grayscale,
        scheme_spacepigs,
        scheme_moody_blues,
        scheme_for_john,
        scheme_gray_orange
};

void pprint_vector(struct vector *v, enum color_scheme scheme);
void pprint_matrix(struct matrix *m, enum color_scheme scheme);

double scale_value(double v, double min, double max);
void value_as_color(double v, enum color_scheme scheme);

#endif /* PPRINT_H */
