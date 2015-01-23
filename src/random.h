/*
 * random.h
 *
 * Copyright 2012-2015 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef RANDOM_H
#define RANDOM_H

#include "matrix.h"
#include "network.h"

/**************************************************************************
 *************************************************************************/
void randomize_gaussian(struct matrix *m, struct network *n);
void randomize_range(struct matrix *m, struct network *n);
void randomize_nguyen_widrow(struct matrix *m, struct network *n);
void randomize_fan_in(struct matrix *m, struct network *n);
void randomize_binary(struct matrix *m, struct network *n);

#endif /* RANDOM_H */
