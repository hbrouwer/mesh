/*
 * act.h
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

#ifndef ACT_H
#define ACT_H

double act_fun_sigmoid(double x);
double act_fun_sigmoid_approx(double x);
double act_fun_sigmoid_deriv(double y);

double act_fun_tanh(double x);
double act_fun_tanh_approx(double x);
double act_fun_tanh_deriv(double y);

double act_fun_linear(double x);
double act_fun_linear_deriv(double y);

#endif /* ACT_H */
