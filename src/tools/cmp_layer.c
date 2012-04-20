/*
 * cmp_layer.c
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

#include <math.h>

#include "cmp_layer.h"

int main(int argc, char **argv)
{
        struct cmp_layer *cl = create_cmp_layer(1, 0.0001, 150, 44);
        init_cmp_layer(cl, 0.5, 1.0);

        cl->input = create_matrix(25000, 44);
        binary_randomize_matrix(cl->input);

        train_layer(cl, 20);

        dispose_cmp_layer(cl);

        exit(EXIT_SUCCESS);
}

struct cmp_layer *create_cmp_layer(double kohonen_lr, double conscience_lr,
                int layer_rows, int layer_cols)
{
        struct cmp_layer *cl;
        if (!(cl = malloc(sizeof(struct cmp_layer))))
                goto error_out;
        memset(cl, 0, sizeof(struct cmp_layer));

        cl->kohonen_lr = kohonen_lr;
        cl->conscience_lr = conscience_lr;

        cl->weights = create_matrix(layer_rows, layer_cols);
        cl->biases = create_vector(layer_rows);

        return cl;

error_out:
        perror("[create_cmp_layer()]");
        return NULL;
}

void dispose_cmp_layer(struct cmp_layer *cl)
{
        dispose_matrix(cl->input);
        dispose_matrix(cl->weights);
        dispose_vector(cl->biases);
        free(cl);
}

void init_cmp_layer(struct cmp_layer *cl, double weight, double bias)
{
        for (int i = 0; i < cl->weights->rows; i++) {
                cl->biases->elements[i] = bias;
                for (int j = 0; j < cl->weights->cols; j++) {
                        cl->weights->elements[i][j] = weight;
                }
        }
}

void train_layer(struct cmp_layer *cl, int max_epochs)
{
        for (int epoch = 1; epoch <= max_epochs; epoch++) {

                printf("Epoch: %d\n", epoch);
                
                for (int i = 0; i < cl->input->rows; i++) {
                        struct vector *iv = row_to_vector(cl->input, i);
                        train_input_vector(cl, iv);
                        dispose_vector(iv);
                }
                
                if(epoch <= 10)
                        cl->kohonen_lr = (1.0 - epoch * .09);
        }
}

void train_input_vector(struct cmp_layer *cl, struct vector *iv)
{
        int shortest_cbd_row = 0;
        double shortest_cbd = 0.0;

        for (int i = 0; i < cl->weights->rows; i++) {
                struct vector *wv = row_to_vector(cl->weights, i);

                double cbd = city_block_distance(iv, wv)
                        - cl->biases->elements[i];
                if (i == 0 || cbd < shortest_cbd) {
                        shortest_cbd_row = i;
                        shortest_cbd = cbd;
                }

                dispose_vector(wv);
        }

        update_layer(cl, iv, shortest_cbd_row);
}

double city_block_distance(struct vector *iv, struct vector *wv)
{
        double cbd = 0.0;
        for (int i = 0; i < iv->size; i++)
                cbd += fabs(wv->elements[i] - iv->elements[i]);

        return cbd;
}

void update_layer(struct cmp_layer *cl, struct vector *iv, 
                int shortest_cbd_row)
{
        for (int i = 0; i < cl->weights->rows; i++) {
                if (i == shortest_cbd_row) {
                        for (int j = 0; j < cl->weights->cols; j++) {
                                double diff = iv->elements[j] - cl->weights->elements[i][j];
                                cl->weights->elements[i][j] += cl->kohonen_lr * diff;
                        }
                        cl->biases->elements[i] += cl->conscience_lr
                                * (1 - cl->biases->elements[i]);
                        if (cl->biases->elements[i] < 1.0) {
                                cl->biases->elements[i] = 1.0;
                        }
                } else {
                        cl->biases->elements[i] += cl->conscience_lr
                                * cl->biases->elements[i];
                }
        }
}
