#!/bin/bash

harbor run \
      --path ./tasks/posttrainbench-gpqamain-qwen3-1.7b \
      --agent codex \
      --model openai/gpt-5.1-codex-max \
      --env modal \
      --ek 'volumes={"/hf-cache-volume":"posttrainbench-hf-cache"}'