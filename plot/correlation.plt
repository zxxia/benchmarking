#!/usr/bin/gnuplot -persist

reset
load 'style.gnu'
set datafile separator ","
set boxwidth 0.4
set style fill solid
set yrange [0:0.8]
set xlabel "Features" #font 'Helvetica,27'
set ylabel "Pearson Coefficient" #font 'Helvetica,27'
# set ylabel "Kendalltau Coefficient" #font 'Helvetica,27'
set output 'correlation/vs_correlation.eps'
plot "vs_correlation.csv" using 1:3:xtic(1) with boxes lc rgb "black" notitle

reset
load 'style.gnu'
set datafile separator ","
set boxwidth 0.4
set style fill solid
set yrange [0:0.8]
set xlabel "Features" #font 'Helvetica,27'
set ylabel "Pearson Coefficient" #font 'Helvetica,27'
# set ylabel "Kendalltau Coefficient" #font 'Helvetica,27'
set output 'correlation/gl_correlation.eps'
plot "gl_correlation.csv" using 1:3:xtic(1) with boxes lc rgb "black" notitle

reset
load 'style.gnu'
set datafile separator ","
set boxwidth 0.4
set style fill solid
set yrange [0:0.8]
set xlabel "Features" #font 'Helvetica,27'
set ylabel "Pearson Coefficient" #font 'Helvetica,27'
# set ylabel "Kendalltau Coefficient" #font 'Helvetica,27'
set output 'correlation/aws_correlation.eps'
plot "aws_correlation.csv" using 1:3:xtic(1) with boxes lc rgb "black" notitle

reset
load 'style.gnu'
set datafile separator ","
set boxwidth 0.4
set style fill solid
# unset xtics
# unset ytics
set yrange [0:0.8]
# set yrange [0:2.0]
# set object 1 rectangle from 0,2 to 1,1 fc 'green' fillstyle solid 0.4 noborder
# set object 2 rectangle from 1,1 to 2,0 fc 'red' fillstyle solid 0.4 noborder
set xlabel "Features" #font 'Helvetica,27'
set ylabel "Pearson Coefficient" #font 'Helvetica,27'
# set ylabel "Kendalltau Coefficient" #font 'Helvetica,27'
set output 'correlation/no_correlation.eps'
plot "no_correlation.csv" using 1:3:xtic(1) with boxes lc rgb "black" notitle

