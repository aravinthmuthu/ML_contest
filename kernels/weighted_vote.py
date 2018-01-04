from collections import defaultdict, Counter
from glob import glob
import sys
import re

glob_files = "results/newbag/*"
loc_outfile = "results/voted_bigbag.csv"
weights_strategy = "weighted"


def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):

  if method == "average":
    scores = defaultdict(list)
  with open(loc_outfile,"wb") as outfile:
    weight_list = [1]*len(glob(glob_files))
    weight_list = [41,39,10,35]
    for i, glob_file in enumerate( glob(glob_files) ):
      print "parsing:", glob_file
      # if weights == "weighted":
         # weight = pattern.match(glob_file)
         # if weight and weight.group(2):
         #    print "Using weight: ",int(weight.group(2))
         #    weight_list[i] = weight_list[i]*int(weight.group(2))
         # else:
         #    print "Using weight: 1"
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      # lines = sorted(lines[:])
      for e, line in enumerate( lines ):
        row = line.strip().split(",")
        for l in range(1,weight_list[i]+1):
          scores[(e,row[0])].append(row[1])
    for j,k in sorted(scores):
      outfile.write("%s,%s\n"%(j,Counter(scores[(j,k)]).most_common(1)[0][0]))
    print("wrote to %s"%loc_outfile)

kaggle_bag(glob_files, loc_outfile, weights=weights_strategy)