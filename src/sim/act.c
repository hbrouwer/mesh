/*
 * act.c
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
#include <stdio.h>

#include "act.h"
#include "vector.h"

/*
 * ########################################################################
 * ## Binary sigmoid function                                            ##
 * ########################################################################
 */

double act_fun_binary_sigmoid(struct vector *v, int i)
{
        double x = v->elements[i];

        return 1.0 / (1.0 + exp(-x));
}

double act_fun_binary_sigmoid_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return y * (1.0 - y);
}

/*
 * ########################################################################
 * ## Bipolar sigmoid function                                           ##
 * ########################################################################
 */

double act_fun_bipolar_sigmoid(struct vector *v, int i)
{
        double x = v->elements[i];

        return (-1.0) + 2.0 / (1.0 + exp(-x));
}

double act_fun_bipolar_sigmoid_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return 0.5 * (1.0 + y) * (1.0 - y);
}

/*
 * ########################################################################
 * ## Softmax function                                                   ##
 * ########################################################################
 */

double act_fun_softmax(struct vector *v, int i)
{
        double x = exp(v->elements[i]);

        double sum = 0.0;
        for (int j = 0; j < v->size; j++)
                sum += exp(v->elements[j]);

        return x / sum;
}

double act_fun_softmax_deriv(struct vector *v, int i)
{
        return 1.0;
}

/*
 * ########################################################################
 * ## Hyperbolic tangent function                                        ##
 * ########################################################################
 */

double act_fun_tanh(struct vector *v, int i)
{
        double x = v->elements[i];

        return tanh(x);
}

double act_fun_tanh_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return 1.0 - y * y;
}

/*
 * ########################################################################
 * ## Linear function                                                    ##
 * ########################################################################
 */

double act_fun_linear(struct vector *v, int i)
{
        double x = v->elements[i];

        return x;
}

double act_fun_linear_deriv(struct vector *v, int i)
{
        return 1.0;
}

/*
 * ########################################################################
 * ## Step function                                                      ##
 * ########################################################################
 */

double act_fun_step(struct vector *v, int i)
{
        double x = v->elements[i];

        if (x >= 0.0)
                return 1.0;
        else
                return 0.0;
}

double act_fun_step_deriv(struct vector *v, int i)
{
        return 1.0;
}
