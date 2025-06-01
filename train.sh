#!/bin/bash
mkdir -p logs
# python pretrain_sweep.py 2>&1 | tee logs/$(date '+%Y-%m-%d_%H%M')_pretrain_sweep.log
python pretrain_full.py 2>&1 | tee logs/$(date '+%Y-%m-%d_%H%M')_pretrain_full.log