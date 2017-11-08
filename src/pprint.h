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

#define VALUE_SYMBOL " "

                /***********************
                 **** color schemes ****
                 ***********************/

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

const static uint32_t PALETTE_BLUE_RED[10] =
        {196, 160, 124, 88,  52,  17,  18,  19,  20,  21};
const static uint32_t PALETTE_BLUE_YELLOW[10] =
        {226, 220, 214, 208, 202, 27,  33,  39,  45,  51};
const static uint32_t PALETTE_GRAYSCALE[10] =
        {255, 253, 251, 249, 247, 245, 243, 241, 239, 237};
const static uint32_t PALETTE_SPACEPIGS[10] =
        {82,  77,  113, 108, 144, 139, 175, 170, 206, 201};
const static uint32_t PALETTE_MOODY_BLUES[10] = 
        {129, 128, 127, 91,  90,  55,  54,  19,  20,  21};
const static uint32_t PALETTE_FOR_JOHN[10] =
        {46,  40,  34,  28,  64,  100, 136, 166, 202, 196};
const static uint32_t PALETTE_GRAY_ORANGE[10] =
        {220, 221, 222, 223, 224, 255, 253, 251, 249, 247};

void pprint_vector(struct vector *v, enum color_scheme scheme);
void pprint_matrix(struct matrix *m, enum color_scheme scheme);

double scale_value(double v, double min, double max);
void value_as_color(double v, enum color_scheme scheme);

#endif /* PPRINT_H */
