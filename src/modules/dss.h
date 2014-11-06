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
void dss_beliefs(struct network *n, struct set *set, struct item *item);
void dss_scores(struct network *n, struct set *set, struct item *item);

/**************************************************************************
 *************************************************************************/
struct matrix *dss_score_matrix(struct network *n, struct set *set,
                struct item *item);

/**************************************************************************
 *************************************************************************/
double dss_comprehension_score(struct vector *a, struct vector *z);
double dss_tau_prior(struct vector *a);
double dss_tau_conditional(struct vector *a, struct vector *z);

/**************************************************************************
 *************************************************************************/
bool is_same_vector(struct vector *a, struct vector *b);

/**************************************************************************
 *************************************************************************/
void dss_word_information(struct network *n, struct item *item);

/**************************************************************************
 *************************************************************************/
struct matrix *dss_word_information_matrix(struct network *n,
                struct item *item);

/**************************************************************************
 *************************************************************************/
int32_t *frequency_table(struct set *s);
void fuzzy_or(struct vector *a, struct vector *b);

#endif /* DSS_H */
