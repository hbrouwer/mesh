/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef CMD_H
#define CMD_H

#include <stdint.h>

#include "network.h"
#include "session.h"

void process_command(char *cmd, struct session *s);

bool cmd_exit(char *cmd, char *fmt, struct session *s);

bool cmd_about(char *cmd, char *fmt, struct session *s);
bool cmd_help(char *cmd, char *fmt, struct session *s);

bool cmd_load_file(char *cmd, char *fmt, struct session *s);

bool cmd_create_network(char *cmd, char *fmt, struct session *s);
bool cmd_remove_network(char *cmd, char *fmt, struct session *s);
bool cmd_networks(char *cmd, char *fmt, struct session *s);
bool cmd_change_network(char *cmd, char *fmt, struct session *s);
bool cmd_inspect(char *cmd, char *fmt, struct session *s);

bool cmd_create_group(char *cmd, char *fmt, struct session *s);
bool cmd_create_bias_group(char *cmd, char *fmt, struct session *s);
bool cmd_remove_group(char *cmd, char *fmt, struct session *s);
bool cmd_groups(char *cmd, char *fmt, struct session *s);
bool cmd_attach_bias(char *cmd, char *fmt, struct session *s);
bool cmd_set_input_group(char *cmd, char *fmt, struct session *s);
bool cmd_set_output_group(char *cmd, char *fmt, struct session *s);
bool cmd_set_act_func(char *cmd, char *fmt, struct session *s);
bool cmd_set_err_func(char *cmd, char *fmt, struct session *s);

bool cmd_create_projection(char *cmd, char *fmt, struct session *s);
bool cmd_remove_projection(char *cmd, char *fmt, struct session *s);
bool cmd_create_elman_projection(char *cmd, char *fmt, struct session *s);
bool cmd_remove_elman_projection(char *cmd, char *fmt, struct session *s);
bool cmd_projections(char *cmd, char *fmt, struct session *s);
bool cmd_projection_table(char *cmd, char *fmt, struct session *s);
bool cmd_freeze_projection(char *cmd, char *fmt, struct session *s);
bool cmd_unfreeze_projection(char *cmd, char *fmt, struct session *s);

bool cmd_toggle_reset_contexts(char *cmd, char *fmt, struct session *s);

#ifdef _OPENMP
bool cmd_toggle_multithreading(char *cmd, char *fmt, struct session *s);
#endif /* _OPENMP */

bool cmd_toggle_pretty_printing(char *cmd, char *fmt, struct session *s);
bool cmd_set_color_scheme(char *cmd, char *fmt, struct session *s);

bool cmd_set_int_parameter(char *cmd, char *fmt, struct session *s);
bool cmd_set_double_parameter(char *cmd, char *fmt, struct session *s);
bool cmd_set_group_double_parameter(char *cmd, char *fmt, struct session *s);

bool cmd_set_random_algorithm(char *cmd, char *fmt, struct session *s);
bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct session *s);
bool cmd_set_update_algorithm(char *cmd, char *fmt, struct session *s);

bool cmd_set_similarity_metric(char *cmd, char *fmt, struct session *s);

bool cmd_set_training_order(char *cmd, char *fmt, struct session *s);

bool cmd_weight_stats(char *cmd, char *fmt, struct session *s);

bool cmd_save_weights(char *cmd, char *fmt, struct session *s);
bool cmd_load_weights(char *cmd, char *fmt, struct session *s);

bool cmd_show_vector(char *cmd, char *fmt, struct session *s);
bool cmd_show_matrix(char *cmd, char *fmt, struct session *s);

bool cmd_load_legacy_set(char *cmd, char *fmt, struct session *s);
bool cmd_load_set(char *cmd, char *fmt, struct session *s);
bool cmd_remove_set(char *cmd, char *fmt, struct session *s);
bool cmd_sets(char *cmd, char *fmt, struct session *s);
bool cmd_change_set(char *cmd, char *fmt, struct session *s);

bool cmd_init(char *cmd, char *fmt, struct session *s);
bool cmd_reset(char *cmd, char *fmt, struct session *s);
bool cmd_train(char *cmd, char *fmt, struct session *s);
bool cmd_test_item(char *cmd, char *fmt, struct session *s);
bool cmd_test_item_num(char *cmd, char *fmt, struct session *s);
bool cmd_test(char *cmd, char *fmt, struct session *s);
bool cmd_test_verbose(char *cmd, char *fmt, struct session *s);

bool cmd_items(char *cmd, char *fmt, struct session *s);
bool cmd_show_item(char *cmd, char *fmt, struct session *s);
bool cmd_show_item_num(char *cmd, char *fmt, struct session *s);

bool cmd_record_units(char *cmd, char *fmt, struct session *s);

bool cmd_set_two_stage_forward(char *cmd, char *fmt, struct session *s);
bool cmd_set_one_stage_forward(char *cmd, char *fmt, struct session *s);
bool cmd_set_two_stage_backward(char *cmd, char *fmt, struct session *s);
bool cmd_set_one_stage_backward(char *cmd, char *fmt, struct session *s);

bool cmd_similarity_matrix(char *cmd, char *fmt, struct session *s);
bool cmd_similarity_stats(char *cmd, char *fmt, struct session *s);
bool cmd_confusion_matrix(char *cmd, char *fmt, struct session *s);
bool cmd_confusion_stats(char *cmd, char *fmt, struct session *s);

bool cmd_create_dcs_group(char *cmd, char *fmt, struct session *s);
bool cmd_dss_test(char *cmd, char *fmt, struct session *s);
bool cmd_dss_scores(char *cmd, char *fmt, struct session *s);
bool cmd_dss_scores_num(char *cmd, char *fmt, struct session *s);
bool cmd_dss_inferences(char *cmd, char *fmt, struct session *s);
bool cmd_dss_inferences_num(char *cmd, char *fmt, struct session *s);
bool cmd_dss_word_info(char *cmd, char *fmt, struct session *s);
bool cmd_dss_word_info_num(char *cmd, char *fmt, struct session *s);
bool cmd_dss_write_word_info(char *cmd, char *fmt, struct session *s);

bool cmd_erp_contrast(char *cmd, char *fmt, struct session *s);
bool cmd_erp_write_values(char *cmd, char *fmt, struct session *s);

bool cmd_tep_test_item(char *cmd, char *fmt, struct session *s);
bool cmd_tep_test_item_num(char *cmd, char *fmt, struct session *s);
bool cmd_tep_record_units(char *cmd, char *fmt, struct session *s);
bool cmd_tep_write_micro_ticks(char *cmd, char *fmt, struct session *s);

                /******************
                 **** commands ****
                 ******************/

#define MAX_FMT_SIZE 48

struct command
{
        /* base command */
        char *cmd_base;
        /* argument format */
        char *cmd_args;
        /* command processor */
        bool (*cmd_proc)(char *cmd, char *fmt, struct session *s);
};

const static struct command cmds[] = {
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"exit",                    NULL,            &cmd_exit},
        {"quit",                    NULL,            &cmd_exit},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"about",                   NULL,            &cmd_about},
        {"help",                    NULL,            &cmd_help},
        {"?",                       NULL,            &cmd_help},
        {"help",                    "%s",            &cmd_help},
        {"?",                       "%s",            &cmd_help},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"loadFile",                "%s",            &cmd_load_file},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"createNetwork",           "%s %s",         &cmd_create_network},
        {"removeNetwork",           "%s",            &cmd_remove_network},
        {"networks",                NULL,            &cmd_networks},
        {"changeNetwork",           "%s",            &cmd_change_network},
        {"inspect",                 NULL,            &cmd_inspect},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"createGroup",             "%s %d",         &cmd_create_group},
        {"createBiasGroup",         "%s",            &cmd_create_bias_group},
        {"createDCSGroup",          "%s %s",         &cmd_create_dcs_group},
        {"removeGroup",             "%s",            &cmd_remove_group},
        {"groups",                  NULL,            &cmd_groups},
        {"attachBias",              "%s",            &cmd_attach_bias},
        {"set InputGroup",          "%s",            &cmd_set_input_group},
        {"set OutputGroup",         "%s",            &cmd_set_output_group},
        {"set ActFunc",             "%s %s",         &cmd_set_act_func},
        {"set ErrFunc",             "%s %s",         &cmd_set_err_func},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"createProjection",        "%s %s",         &cmd_create_projection},
        {"removeProjection",        "%s %s",         &cmd_remove_projection},
        {"createElmanProjection",   "%s %s",         &cmd_create_elman_projection},
        {"removeElmanProjection",   "%s %s",         &cmd_remove_elman_projection},
        {"projections",             NULL,            &cmd_projections},
        {"projectionTable",         NULL,            &cmd_projection_table},
        {"freezeProjection",        "%s %s",         &cmd_freeze_projection},
        {"unfreezeProjection",      "%s %s",         &cmd_unfreeze_projection},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"toggleResetContexts",     NULL,            &cmd_toggle_reset_contexts},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#ifdef _OPENMP
        {"toggleMultithreading",    NULL,            &cmd_toggle_multithreading},
#endif /* _OPENMP */

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"togglePrettyPrinting",    NULL,            &cmd_toggle_pretty_printing},
        {"set ColorScheme",         "%s",            &cmd_set_color_scheme},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        /*
         * Int parameters: BatchSize, MaxEpochs, ReportAfter, RandomSeed,
         *      BackTicks;
         */
        {"set",                     "%s %d",         &cmd_set_int_parameter},

        /*
         * Double parameters: InitContextUnits, RandomMu, RandomSigma,
         *      RandomMin, RandomMax, LearningRate, LRScaleFactor,
         *      LRScaleAfter, Momentum, MNScaleFactor, MNScaleAfter,
         *      WeightDecay, WDScaleFactor, WDScaleAfter, ErrorThreshold,
         *      TargetRadius, ZeroErrorRadius, RpropInitUpdate,
         *      RpropEtaPlus, RpropEtaMinus, DBDRateIncrement,
         *      DBDRateDecrement;
         */
        {"set",                     "%s %lf",        &cmd_set_double_parameter},

        /* 
         * Group double parameters: ReLUAlpha, ReLUMaxValue, LogisticFSC,
         *      LogisticGain;
         */
        {"set",                     "%s %s %lf",     &cmd_set_group_double_parameter},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"set RandomAlgorithm",     "%s"  ,          &cmd_set_random_algorithm},
        {"set LearningAlgorithm",   "%s",            &cmd_set_learning_algorithm},
        {"set UpdateAlgorithm",     "%s",            &cmd_set_update_algorithm},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"set SimilarityMetric",    "%s",            &cmd_set_similarity_metric},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"set TrainingOrder",       "%s",            &cmd_set_training_order},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"weightStats",             NULL,            &cmd_weight_stats},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"saveWeights",             "%s",            &cmd_save_weights},
        {"loadWeights",             "%s",            &cmd_load_weights},

        /*
         * Vector types: units, error;
         */
        {"showVector",              "%s %s",         &cmd_show_vector},

        /*
         * Matrix types: weights, gradients, dynamics;
         */
        {"showMatrix",              "%s %s %s",      &cmd_show_matrix},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"loadLegacySet",           "%s %s",         &cmd_load_legacy_set},
        {"loadSet",                 "%s %s",         &cmd_load_set},
        {"removeSet",               "%s",            &cmd_remove_set},
        {"sets",                    NULL,            &cmd_sets},
        {"changeSet",               "%s",            &cmd_change_set},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"init",                    NULL,            &cmd_init},
        {"reset",                   NULL,            &cmd_reset},
        {"train",                   NULL,            &cmd_train},
        {"testItem",                "\"%[^\"]\"",    &cmd_test_item},
        {"testItem",                "'%[^']'",       &cmd_test_item},
        {"testItem",                "%d",            &cmd_test_item_num},
        {"test",                    NULL,            &cmd_test},
        {"testVerbose",             NULL,            &cmd_test_verbose},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"items",                   NULL,            &cmd_items},
        {"showItem",                "\"%[^\"]\"",    &cmd_show_item},
        {"showItem",                "'%[^']'",       &cmd_show_item},
        {"showItem",                "%d",            &cmd_show_item_num},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"recordUnits",             "%s %s",         &cmd_record_units},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"set TwoStageForward",     "%s %s",         &cmd_set_two_stage_forward},
        {"set OneStageForward",     NULL,            &cmd_set_one_stage_forward},
        {"set TwoStageBackward",    "%s %s",         &cmd_set_two_stage_backward},
        {"set OneStageBackward",    NULL,            &cmd_set_one_stage_backward},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"similarityMatrix",        NULL,            &cmd_similarity_matrix},
        {"similarityStats",         NULL,            &cmd_similarity_stats},
        {"confusionMatrix",         NULL,            &cmd_confusion_matrix},
        {"confusionStats",          NULL,            &cmd_confusion_stats},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"dssTest",                 NULL,            &cmd_dss_test},
        {"dssScores",               "%s \"%[^\"]\"", &cmd_dss_scores},
        {"dssScores",               "%s '%[^']'",    &cmd_dss_scores},
        {"dssScores",               "%s %d",         &cmd_dss_scores_num},
        {"dssInferences",           "%s \"%[^\"]\" %lf",
                                                     &cmd_dss_inferences},
        {"dssInferences",           "%s '%[^']' %lf",
                                                     &cmd_dss_inferences},
        {"dssInferences",           "%s %d %lf",
                                                     &cmd_dss_inferences_num},                                                     
        {"dssWordInfo",             "%s \"%[^\"]\"", &cmd_dss_word_info},
        {"dssWordInfo",             "%s '%[^']'",    &cmd_dss_word_info},
        {"dssWordInfo",             "%s %d",         &cmd_dss_word_info_num},
        {"dssWriteWordInfo",        "%s %s",         &cmd_dss_write_word_info},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"erpContrast",             "%s \"%[^\"]\" \"%[^\"]\"",
                                                     &cmd_erp_contrast},
        {"erpContrast",             "%s '%[^']' \"%[^\"]\"",
                                                     &cmd_erp_contrast},
        {"erpContrast",             "%s \"%[^\"]\" '%[^']'",
                                                     &cmd_erp_contrast},
        {"erpContrast",             "%s '%[^']' '%[^']'",
                                                     &cmd_erp_contrast},
        {"erpWriteValues",          "%s %s %s",      &cmd_erp_write_values},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"tepTestItem",             "%s %lf %lf \"%[^\"]\"",
                                                     &cmd_tep_test_item},
        {"tepTestItem",             "%s %lf %lf '%[^']'",
                                                     &cmd_tep_test_item},
        {"tepTestItem",             "%s %lf %lf %d",
                                                     &cmd_tep_test_item_num},
        {"tepRecordUnits",          "%s %lf %lf %s %s",
                                                     &cmd_tep_record_units},
        {"tepWriteMicroTicks",      "%s %lf %lf %s",
                                                     &cmd_tep_write_micro_ticks},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {NULL,                      NULL,            NULL} /* tail */
};

#endif /* CMD_H */
