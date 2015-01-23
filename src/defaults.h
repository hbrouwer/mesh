/*
 * defaults.h
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

#ifndef DEFAULTS_H
#define DEFAULTS_H

#define DFLT_RANDOM_ALGORITHM   randomize_range
#define DFLT_RANDOM_MU          0.0
#define DFLT_RANDOM_SIGMA       0.5
#define DFLT_RANDOM_MIN         -1.0
#define DFLT_RANDOM_MAX         +1.0

#define DFLT_LEARNING_RATE      0.05
#define DFLT_LR_SCALE_FACTOR    0.0
#define DFLT_LR_SCALE_AFTER     0.0

#define DFLT_MOMENTUM           0.4
#define DFLT_MN_SCALE_FACTOR    0.0
#define DFLT_MN_SCALE_AFTER     0.0

#define DFLT_WEIGHT_DECAY       0.0
#define DFLT_WD_SCALE_AFTER     0.0
#define DFLT_WD_SCALE_FACTOR    0.0

#define DFLT_TARGET_RADIUS      0.0
#define DFLT_ZERO_ERROR_RADIUS  0.1

#define DFLT_ERROR_THRESHOLD    0.05
#define DFLT_MAX_EPOCHS         1000
#define DFLT_REPORT_AFTER       100

#define DFLT_RP_INIT_UPDATE     0.0125
#define DFLT_RP_ETA_PLUS        1.2
#define DFLT_RP_ETA_MINUS       0.5

#define DFLT_DBD_RATE_INCREMENT 0.1
#define DFLT_DBD_RATE_DECREMENT 0.9

#define DFLT_SIMILARITY_METRIC  cosine

#endif /* DEFAULTS_H */
