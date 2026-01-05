#!/usr/bin/env bash
find versions/v3-stacked-lstm -maxdepth 2 -type f -printf '%p %TY-%Tm-%Td %TH:%TM:%TS\n'