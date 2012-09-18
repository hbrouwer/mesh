/*
 * main.c
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

#include <stdarg.h>

#include "ffn_unfold.h"

#include "erps.h"
#include "main.h"
#include "network.h"
#include "train.h"

int main(int argc, char **argv)
{
        struct network *n = NULL;
        bool net_spec = false;

        cprintf("");
        mprintf("MESH %s", VERSION);
        mprintf("(c) 2012 Harm Brouwer <me@hbrouwer.eu>");
        mprintf("Center for Language and Cognition, University of Groningen &");
        mprintf("Netherlands Organisation for Scientific Research (NWO)");
        cprintf("");

        for (int i = 1; i < argc; i++) {
                if (strcmp(argv[i], "--network") == 0) {
                        if (++i < argc) {
                                n = load_network(argv[i]);
                                net_spec = true;
                        }
                }

                if (strcmp(argv[i], "--save_weights") == 0) {
                        if (++i < argc) {
                                n->weights_file = argv[i];
                                n->save_weights = true;
                        }
                }

                if (strcmp(argv[i], "--load_weights") == 0) {
                        if (++i < argc) {
                                n->weights_file = argv[i];
                                n->load_weights = true;
                        }
                }

                if (strcmp(argv[i], "--help") == 0) {
                        print_help(argv[0]);
                        goto exit_success;
                }

                if (strcmp(argv[i], "--version") == 0) {
                        print_version();
                        goto exit_success;
                }

        }

        if (!net_spec) {
                print_help(argv[0]);
                goto exit_success;
        }

        if (!n)
                goto exit_success;

        initialize_network(n);
        if (!n->load_weights)
                train_network(n);
        if (n->learning_algorithm == train_bp) {
                test_network(n);
                // compute_erp_correlates(n);
        } else {
                test_unfolded_network(n);
        }

        if (n->save_weights)
                save_weights(n);

        mprintf("Cleaning up...");
        dispose_network(n);

exit_success:
        exit(EXIT_SUCCESS);
}

void print_help(char *exec_name)
{
        cprintf(
                        "usage: %s [options]\n\n"

                        "  running network simulations:\n"
                        "    --network <file>\t\tload and test the network specified in <file>\n"
                        "    --save_weights <file>\tsave weight matrices to <file> after training\n"
                        "    --load_weights <file>\tload weight matrices from <file>\n"

                        "\n"
                        "  basic information for users:\n"
                        "    --help\t\t\tshows this help message\n"
                        "    --version\t\t\tshows version\n",
                        exec_name);
}

void print_version()
{
        cprintf("%s\n", VERSION);
}

void cprintf(const char *fmt, ...)
{
        va_list args;

        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

void mprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "--- ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

void eprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "!!! ERROR: ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

void pprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "=== ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

void rprintf(const char *fmt, ...)
{
        va_list args;

        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
        fprintf(stdout, "\n");
}
