#!/usr/bin/env python3

import sys
import json

MAX_LINES = 100000000
num_lines = 0
for line in sys.stdin:
    try:
        data = json.loads(line)
        print(json.dumps(data))
        num_lines += 1
        if num_lines >= MAX_LINES:
            break
    except:
        continue
