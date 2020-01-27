#!/usr/bin/gnuplot -persist
reset
load 'style.gnu'
set datafile separator ","
# set term x11 0
# set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "CDF"
set xlabel "Compute cost(GPU)"
set output 'cost_dist/vs_cost_dist.eps'
set label "90% videos have cost savings <= ??%" at 0.3, 0.2 font 'Helvetica,27'
set label "??% videos have cost savings <= 50%" at 0.3, 0.3 font 'Helvetica,27'
plot "vs_cdf.csv" using 1:2 pt 7 ps 0.8 lt rgb 'red' notitle


reset
# set term x11 1
load 'style.gnu'
set datafile separator ","
# set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "CDF"
set xlabel "Compute cost(GPU)"
set output 'cost_dist/gl_cost_dist.eps'
set label "90% videos have cost savings <= 90%" at 0.3, 0.2 font 'Helvetica,27'
set label "??% videos have cost savings <= 50%" at 0.3, 0.3 font 'Helvetica,27'
plot "gl_cdf.csv" using 1:2 pt 7 ps 0.8 lt rgb 'red' notitle


# set term x11 2
reset
load 'style.gnu'
set datafile separator ","
# set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "CDF"
set xlabel "Network Cost(Bandwidth)"
set output 'cost_dist/aw_cost_dist.eps'
set label "90% videos have cost savings <= 90%" at 0.3, 0.2 font 'Helvetica,27'
set label "??% videos have cost savings <= 50%" at 0.3, 0.3 font 'Helvetica,27'
plot "aw_cdf.csv" using 1:2 pt 7 ps 0.8 lt rgb 'red' notitle


# set term x11 3
reset
load 'style.gnu'
set datafile separator ","
# set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "CDF"
set xlabel "Compute Cost(GPU)"
set output 'cost_dist/no_cost_dist.eps'
set label "90% videos have cost savings <= 90%" at 0.3, 0.2 font 'Helvetica,27'
set label "??% videos have cost savings <= 50%" at 0.3, 0.3 font 'Helvetica,27'
plot "no_cdf.csv" using 1:2 pt 7 ps 0.8 lt rgb 'red' notitle
