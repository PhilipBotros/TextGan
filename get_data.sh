#!/bin/bash

mkdir data 
wget http://research.signalmedia.co/newsir16/signalmedia-1m.jsonl.gz
gzip -d signalmedia-1m.jsonl.gz
mv signalmedia-1m.jsonl ./data/sample-1m.jsonl