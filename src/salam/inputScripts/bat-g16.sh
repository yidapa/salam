#!/bin/bash
for  inf  in  cpd*.com
do

newname=`echo  ${inf#} |  sed 's/.com//g' `  
echo ${newname}  
Modify_g09_pbs_2.sh    $inf     ${newname}    g16.pbs.sh 
qsub  g16.pbs.sh

done

