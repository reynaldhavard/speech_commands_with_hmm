#!/bin/bash

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir data
tar -xvzf speech_commands_v0.01.tar.gz -C data/
rm speech_commands_v0.01.tar.gz
