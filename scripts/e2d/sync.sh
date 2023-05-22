#!/bin/bash

rsync -chavP --stats --progress --exclude="*run-*" --exclude="*/_*" --exclude="_*" jkirschn@beluga.computecanada.ca:/home/jkirschn/scratch/jkirschn/runs /home/johannes/Documents/PhD/Projects/ids-pm/code/runs-e2d