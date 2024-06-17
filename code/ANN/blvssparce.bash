#!/bin/bash -f

echo "bloom"
python ANN_forward.py -s ../../data/ANN/A0201_bl.syn -e ../../data/ANN/A0201_evaluation -bl | grep -v "#" | gawk '{print $2,$3}' | ./xycorr
echo ""

echo "sparce"

python ANN_forward.py -s ../../data/ANN/A0201_sp.syn -e ../../data/ANN/A0201_evaluation | grep -v "#" | gawk '{print $2,$3}' | ./xycorr

echo ""

echo "averaged"

python ANN_forward.py -s ../../data/ANN/A0201_bl.syn -e ../../data/ANN/A0201_evaluation -bl | grep -v "#" > A0201_bl.pred
python ANN_forward.py -s ../../data/ANN/A0201_sp.syn -e ../../data/ANN/A0201_evaluation | grep -v "#" > A0201_sp.pred
paste A0201_sp.pred A0201_bl.pred | gawk '{print $2,($3+$6)/2}' | ./xycorr