/*
 * pprint.c
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

#include "main.h"
#include "pprint.h"

#define VALUE_SYMBOL "  "

#define SCHEME_BLUE_RED 0
#define SCHEME_BLUE_YELLOW 1
#define SCHEME_GRAYSCALE 2

void pprint_vector(struct vector *v)
{
        double min = v->elements[0];
        double max = v->elements[0];
        
        for (int i = 1; i < v->size; i++) {
                if (v->elements[i] < min)
                        min = v->elements[i];
                if (v->elements[i] > max)
                        max = v->elements[i];
        }

        for (int i = 0; i < v->size; i++) {
               
                /*
                 * Scale value into [0,1] interval.
                 */
                double val;
                if (max > min)
                        val = (v->elements[i] - min) / (max - min);
                /*
                 * If we are dealing with a one-unit vector, or with a
                 * vector of which all units have the same value, we
                 * somehow have to determine whether this value is high
                 * or low.
                 *
                 * If value is in the interval [0,1], the scaled value
                 * is simply the original value:
                 */
                else if (v->elements[i] >= 0.0 && v->elements[i] <= 1.0)
                        val = v->elements[i];
                /*
                 * If value is in the interval [-1,1], we scale the value
                 * into the [0,1] interval.
                 */
                else if (v->elements[i] >= -1.0 && v->elements[i] <= 1.0)
                        val = (v->elements[i] + 1.0) / 2.0;

                pprint_value_as_color(SCHEME_BLUE_YELLOW, val);
        }
        printf("\n");
}

void pprint_value_as_color(int scheme, double v)
{
        if (scheme == SCHEME_BLUE_RED)
                pprint_value_scheme_blue_red(v);
        if (scheme == SCHEME_BLUE_YELLOW)
                pprint_value_scheme_blue_yellow(v);
        if (scheme == SCHEME_GRAYSCALE)
                pprint_value_scheme_grayscale(v);
}

void pprint_value_scheme_blue_red(double v)
{
        if (v >= 0.90)
                printf("\x1b[48;05;196m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.80 && v < 0.90)
                printf("\x1b[48;05;160m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.70 && v < 0.80)
                printf("\x1b[48;05;124m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.60 && v < 0.70)
                printf("\x1b[48;05;088m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.50 && v < 0.60)
                printf("\x1b[48;05;052m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.40 && v < 0.50)
                printf("\x1b[48;05;017m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.30 && v < 0.40)
                printf("\x1b[48;05;018m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.20 && v < 0.30)
                printf("\x1b[48;05;019m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.10 && v < 0.20)
                printf("\x1b[48;05;020m%s\x1b[0m", VALUE_SYMBOL);
        if (v < 0.10)
                printf("\x1b[48;05;021m%s\x1b[0m", VALUE_SYMBOL);
}

void pprint_value_scheme_blue_yellow(double v)
{
        if (v >= 0.90)
                printf("\x1b[48;05;226m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.80 && v < 0.90)
                printf("\x1b[48;05;220m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.70 && v < 0.80)
                printf("\x1b[48;05;214m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.60 && v < 0.70)
                printf("\x1b[48;05;208m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.50 && v < 0.60)
                printf("\x1b[48;05;202m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.40 && v < 0.50)
                printf("\x1b[48;05;027m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.30 && v < 0.40)
                printf("\x1b[48;05;033m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.20 && v < 0.30)
                printf("\x1b[48;05;039m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.10 && v < 0.20)
                printf("\x1b[48;05;045m%s\x1b[0m", VALUE_SYMBOL);
        if (v < 0.10)
                printf("\x1b[48;05;051m%s\x1b[0m", VALUE_SYMBOL);
}

void pprint_value_scheme_grayscale(double v)
{
        if (v >= 0.90)
                printf("\x1b[48;05;255m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.80 && v < 0.90)
                printf("\x1b[48;05;253m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.70 && v < 0.80)
                printf("\x1b[48;05;251m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.60 && v < 0.70)
                printf("\x1b[48;05;249m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.50 && v < 0.60)
                printf("\x1b[48;05;247m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.40 && v < 0.50)
                printf("\x1b[48;05;245m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.30 && v < 0.40)
                printf("\x1b[48;05;243m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.20 && v < 0.30)
                printf("\x1b[48;05;241m%s\x1b[0m", VALUE_SYMBOL);
        if (v >= 0.10 && v < 0.20)
                printf("\x1b[48;05;239m%s\x1b[0m", VALUE_SYMBOL);
        if (v < 0.10)
                printf("\x1b[48;05;237m%s\x1b[0m", VALUE_SYMBOL);
} 
