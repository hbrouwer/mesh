#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2022 Harm Brouwer <me@hbrouwer.eu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Extract relevant topics from the 'src/help.h' file, and write each topic 
# into a Markdown file. Cross-references topics will be turned into links.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_help_header():
    with open("../src/help.h", "r") as f:
        lines = f.readlines()
    return lines

# Compile a list of relevant topics.
def parse_topics(lines):
    topics = dict()
    for line in lines:
        topic = re.search("#define TOPIC_(.+) ", line)
        if (topic):
            topic = topic.group(1)
            # skip 'ABOOT'
            if topic != "ABOOT":
                topics[topic] = []
    return topics

# For each relevant topic, extract the help text, and add links for
# cross-references.
def parse_topic_texts(lines, topics):
    for i in range(0, len(lines)):
        topic = re.search("#define TOPIC_(.+) ", lines[i])
        if (topic and topic.group(1) in topics.keys()):
            topic = topic.group(1)
            # extract topic body
            while True:
                i = i + 1
                line = re.search(r'"(.+)\\n" \\', lines[i])
                if line:
                    line = parse_xref(line.group(1), topics)
                    topics[topic].append(line)
                else:
                    break

# Cross-references are contained within square brackets [topic], and should
# occur in the list of relevant topics. If this is the case, they will be
# instantiated as a link
def parse_xref(line, topics):
    xref = re.search('\[(.+)]', line)
    if (xref and xref.group(1).upper() in topics.keys()):
        xref = xref.group(1)
        line = re.sub("\[" + xref + "\]", "[" + xref + "](" + xref + ".md)", line)
    return line

# Write each help topic to a seperate Markdown file.
def write_markdown(topics):
    for topic in topics.keys():
        with open (topic.lower() + ".md", "w") as f:
            lines = topics[topic]
            # the 'ABOUT' topic is special
            if topic == 'ABOUT':
                f.write("```\n")
                for line in lines:
                    f.write(line + "\n")
                f.write("```\n")
            else:
                for i in range(0, len(lines)):
                    # strip all leading and trailing whitespace, 
                    f.write(lines[i].strip() + "\n")
                    # if a next line exists, and does not start with a 
                    # whitespace, add a newline for proper formatting
                    if (i + 1 < len(lines) and not re.match("^\s", lines[i + 1])):
                        f.write("\n")

if __name__ == "__main__":
    lines = read_help_header()
    topics = parse_topics(lines)
    parse_topic_texts(lines, topics)
    write_markdown(topics)
