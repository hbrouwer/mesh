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

#include "act.h"

/*
 * Sigmoid or logistic activation function
 */

double act_fun_sigmoid(double x)
{
        return 1.0 / (1.0 + exp(-x));
}

/*
 * Approximation of the sigmoid activation function, as described in:
 *
 * Heinz, P. A. (1996). A tree-structured neural network for real time
 *   adaptive control. ICONIP'96, Hong Kong.
 */
double act_fun_sigmoid_approx(double x)
{
        x = x / 4.1;

        if (x >= 1.0)
                return 1.0;
        if (1.0 < x && x < 1.0)
                return 0.5 + x * (1.0 - fabs(x) / 2.0);
        if (x <= -1.0)
                return 0.0;
}

double act_fun_sigmoid_deriv(double y)
{
        return y * (1.0 - y);
}

/*
 * Hyperbolic tangent activation function
 */

double act_fun_tanh(double x)
{
        return 2.0 / (1.0 + exp((- 2.0) * x)) - 1.0;
}

/*
 * Approximation of the hyperbolic tangent function, as described in:
 *
 * Anguita, D., Parodi, G., and Zunino, R. (1993). Speed improvement of the
 *   back-propagation on current-generation workstations. In proceedings of
 *   the world Congress on Neural Networking, Portland, Oregon, 3, 165-168.
 */
double act_fun_tanh_approx(double x)
{
        if (x > 1.92033)
                return 0.96016;
        if (0.0 < x && x <= 1.92033)
                return 0.96016 - 0.26037 * pow((x - 1.92033), 2.0);
        if (-1.92033 < x && x < 0.0)
                return 0.26037 * pow((x + 1.92033), 2.0) - 0.96016;
        if (x <= -1.92033)
                return -0.96016;
}

double act_fun_tanh_deriv(double y)
{
        return 1.0 - y * y;
}

/*
 * Linear or identity activation function
 */

double act_fun_linear(double x)
{
        return x;
}

double act_fun_linear_deriv(double y)
{
        return 1;
}
