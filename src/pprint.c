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

#include "main.h"
#include "pprint.h"

void pprint_vector(struct vector *v, enum color_scheme scheme)
{
        double min = vector_minimum(v);
        double max = vector_maximum(v);
        if (min > 0.0)
                min = 0.0;
        for (uint32_t i = 0; i < v->size; i++) {
                double sv = scale_value(v->elements[i], min, max);
                value_as_color(sv, scheme);
        }
        cprintf("\n");
}

void pprint_matrix(struct matrix *m, enum color_scheme scheme)
{
        double min = matrix_minimum(m);
        double max = matrix_maximum(m);
        if (min > 0.0)
                min = 0.0;
        for (uint32_t i = 0; i < m->rows; i++) {
                for (uint32_t j = 0; j < m->cols; j++) {
                        double sv = scale_value(m->elements[i][j], min, max);
                        value_as_color(sv, scheme);
                }
                cprintf("\n");
        }
}

double scale_value(double v, double min, double max)
{
        double sv = 0.0;

        /*
         * Scale value into [0,1] interval.
         */
        if (max > min)
                sv = (v - min) / (max - min);
        /*
         * If we are dealing with a one-unit vector, or with a vector of
         * which all units have the same value, we somehow have to determine
         * whether this value is high or low.
         *
         * If value is in the interval [0,1], the scaled value is simply the
         * original value:
         */
        else if (v >= 0.0 && v <= 1.0)
                sv = v;
        /*
         * If value is in the interval [-1,1], we scale the value into the
         * [0,1] interval.
         */
        else if (v >= -1.0 && v <= 1.0)
                sv = (v + 1.0) / 2.0;

        return sv;
}

void value_as_color(double v, enum color_scheme scheme)
{
        const uint32_t *palette;

        switch (scheme) {
        case scheme_blue_red:
                palette = PALETTE_BLUE_RED;
                break;
        case scheme_blue_yellow:
                palette = PALETTE_BLUE_YELLOW;
                break;
        case scheme_grayscale:
                palette = PALETTE_GRAYSCALE;
                break;
        case scheme_spacepigs:
                palette = PALETTE_SPACEPIGS;
                break;
        case scheme_moody_blues:
                palette = PALETTE_MOODY_BLUES;
                break;
        case scheme_for_john:
                palette = PALETTE_FOR_JOHN;
                break;
        case scheme_gray_orange:
                palette = PALETTE_GRAY_ORANGE;
                break;
        default:
                palette = PALETTE_BLUE_RED;
        }

        if (v >= 0.90)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[0], VALUE_SYMBOL);
        if (v >= 0.80 && v < 0.90)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[1], VALUE_SYMBOL);
        if (v >= 0.70 && v < 0.80)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[2], VALUE_SYMBOL);
        if (v >= 0.60 && v < 0.70)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[3], VALUE_SYMBOL);
        if (v >= 0.50 && v < 0.60)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[4], VALUE_SYMBOL);
        if (v >= 0.40 && v < 0.50)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[5], VALUE_SYMBOL);
        if (v >= 0.30 && v < 0.40)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[6], VALUE_SYMBOL);
        if (v >= 0.20 && v < 0.30)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[7], VALUE_SYMBOL);
        if (v >= 0.10 && v < 0.20)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[8], VALUE_SYMBOL);
        if (v < 0.10)
                cprintf("\x1b[48;05;%dm%s\x1b[0m", palette[9], VALUE_SYMBOL);
}
