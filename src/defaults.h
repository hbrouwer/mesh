/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef DEFAULTS_H
#define DEFAULTS_H

#define DEFAULT_RESET_CONTEXTS     true
#define DEFAULT_INIT_CONTEXT_UNITS 0.5
#define DEFAULT_PRETTY_PRINTING    true
#define DEFAULT_COLOR_SCHEME       scheme_blue_yellow
#define DEFAULT_RANDOM_ALGORITHM   randomize_range
#define DEFAULT_RANDOM_MU          0.0
#define DEFAULT_RANDOM_SIGMA       0.5
#define DEFAULT_RANDOM_MIN         -1.0
#define DEFAULT_RANDOM_MAX         +1.0
#define DEFAULT_LEARNING_ALGORITHM train_network_with_bp
#define DEFAULT_UPDATE_ALGORITHM   bp_update_sd
#define DEFAULT_LEARNING_RATE      0.05
#define DEFAULT_LR_SCALE_FACTOR    0.0
#define DEFAULT_LR_SCALE_AFTER     0.0
#define DEFAULT_MOMENTUM           0.4
#define DEFAULT_MN_SCALE_FACTOR    0.0
#define DEFAULT_MN_SCALE_AFTER     0.0
#define DEFAULT_WEIGHT_DECAY       0.0
#define DEFAULT_WD_SCALE_AFTER     0.0
#define DEFAULT_WD_SCALE_FACTOR    0.0
#define DEFAULT_TARGET_RADIUS      0.0
#define DEFAULT_ZERO_ERROR_RADIUS  0.0
#define DEFAULT_ERROR_THRESHOLD    0.05
#define DEFAULT_MAX_EPOCHS         1000
#define DEFAULT_REPORT_AFTER       100
#define DEFAULT_RP_INIT_UPDATE     0.0125
#define DEFAULT_RP_ETA_PLUS        1.2
#define DEFAULT_RP_ETA_MINUS       0.5
#define DEFAULT_DBD_RATE_INCREMENT 0.1
#define DEFAULT_DBD_RATE_DECREMENT 0.9
#define DEFAULT_SIMILARITY_METRIC  cosine
#define DEFAULT_RELU_ALPHA         0.1
#define DEFAULT_RELU_MAX           INFINITY
#define DEFAULT_LOGISTIC_FSC       0.1
#define DEFAULT_LOGISTIC_GAIN      1.0

#endif /* DEFAULTS_H */
