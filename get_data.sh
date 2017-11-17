#!/bin/bash

mkdir data && cd data
wget http://research.signalmedia.co/newsir16/signalmedia-1m.jsonl.gz
gzip -d signalmedia-1m.jsonl.gz