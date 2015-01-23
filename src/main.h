/*
 * main.h
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

#ifndef MAIN_H
#define MAIN_H

#define MAX_ARG_SIZE 128
#define MAX_BUF_SIZE 32768

/**************************************************************************
 *************************************************************************/
#ifdef _OPENMP
void print_openmp_status();
#endif /* _OPENMP */

/**************************************************************************
 *************************************************************************/
void print_welcome();
void print_help(char *exec_name);

/**************************************************************************
 *************************************************************************/
void cprintf(const char *fmt, ...);
void mprintf(const char *fmt, ...);
void eprintf(const char *fmt, ...);
void pprintf(const char *fmt, ...);

#endif /* MAIN_H */
