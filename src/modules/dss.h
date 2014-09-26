/*
 * dss.h
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef DSS_H
#define DSS_H

#include "../network.h"

/**************************************************************************
 *************************************************************************/
void dss_test(struct network *n);
void dss_test_item(struct network *n, struct item *item);
void dss_beliefs(struct network *n, struct set *set, struct item *item);

/**************************************************************************
 *************************************************************************/
double dss_comprehension_score(struct vector *a, struct vector *z);

/**************************************************************************
 *************************************************************************/
double dss_tau_prior(struct vector *a);
double dss_tau_conditional(struct vector *a, struct vector *z);

/**************************************************************************
 *************************************************************************/
void dss_surprisal(struct network *n, struct item *item);
double dss_syntactic_surprisal(struct set *s, struct item *item,
                uint32_t word);
double dss_semantic_surprisal(struct vector *pv, struct vector *v);

#endif /* DSS_H */
