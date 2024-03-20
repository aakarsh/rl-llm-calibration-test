#!/bin/bash

docker exec -it $(docker ps | grep aakarsh/llm_calibration | awk '{print $1}') bash
