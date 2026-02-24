#!/bin/bash

echo "1 2 6 8 11" | tr -s " " "\n" | xargs -I %1 -n 1 -P 5 script/step2.sh "zheer_zr%1_1234"