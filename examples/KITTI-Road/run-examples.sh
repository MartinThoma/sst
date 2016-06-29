#!/usr/bin/env bash
time sst train --hypes fully_simple_road.json
time sst test --hypes fully_simple_road.json --out out
time sst train --hypes sliding_window_road.json