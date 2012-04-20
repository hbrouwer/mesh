/*
 * cmp_layer.h
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

#ifndef CMP_LAYER_H
#define CMP_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../sim/matrix.h"
#include "../sim/vector.h"

struct cmp_layer
{
        struct matrix *input;       /* input "vectors" */
        struct matrix 
                *weights;
        struct vector
                *biases;

        double kohonen_lr;          /* learning rate for Kohonen weights */
        double conscience_lr;       /* learning rate for conscience bias */
};

struct cmp_layer *create_cmp_layer(double kohonen_lr, double conscience_lr,
                int layer_rows, int layer_cols);
void dispose_cmp_layer(struct cmp_layer *cl);
void init_cmp_layer(struct cmp_layer *cl, double weight, double bias);

void train_layer(struct cmp_layer *cl, int max_epochs);
void train_input_vector(struct cmp_layer *cl, struct vector *iv);

double city_block_distance(struct vector *iv, struct vector *wv);
void update_layer(struct cmp_layer *cl, struct vector *iv, 
                int shortest_cbd_row);

#endif /* CMP_LAYER_H */
