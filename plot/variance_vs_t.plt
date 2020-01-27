#!/usr/bin/gnuplot -persist
reset
load 'style.gnu'
set datafile separator ","
set xtics
set ytics
set xlabel "Time(second)" # font 'Helvetica,27'
set ylabel "Accuracy" # font 'Helvetica,27'
set output 'var_over_time/acc_over_time.eps'
# set xrange [0:2.0]
set yrange [0.7:1.1]
# show style line
set key right bottom font 'Helvetica,30'
plot "var_over_time.csv" using 1:2  pt 7 lw 4 lc rgb 'red' with lines title 'VideoStorm',\
     "var_over_time.csv" using 1:4 dt 3 pt 7 lw 4 lc rgb 'blue' with lines title 'Glimpse'


reset
load 'style.gnu'
set datafile separator ","
set yrange [0:0.3]
set xtics
set ytics
set xlabel "Time(second)" # font 'Helvetica,27'
set ylabel "Cost" # font 'Helvetica,27'
set key right top font 'Helvetica,30'
set output 'var_over_time/cost_over_time.eps'
# set label "VideoStorm strictly\nbetter than Glimpse" at 0.2, 1.9 font 'Helvetica,20'
# set label "VideoStorm strictly\nworse than Glimpse" at 1.2, 0.2 font 'Helvetica,20'
plot "var_over_time.csv" using 1:3 pt 7 ps 1 lw 4 lc rgb 'red' with lines title 'VideoStorm',\
     "var_over_time.csv" using 1:5 dt 3 pt 7 lw 4 lc rgb 'blue' with lines title 'Glimpse'
