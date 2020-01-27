#!/usr/bin/gnuplot -persist

# reset
# load 'style.gnu'
# set datafile separator ","
# set boxwidth 0.4
# set style fill solid
# set yrange [-1:1]
# set xlabel "Features" font 'Helvetica,27'
# set ylabel "Pearson Coefficient" font 'Helvetica,27'
# set output 'correlation/vs_correlation.eps'
# plot "vs_correlation.csv" using 1:3:xtic(1) with boxes lc rgb "black" notitle


reset
load 'style.gnu'
set datafile separator ","
set yrange [0:1]
set xrange [0:0.15]

set xlabel "GPU cycles(normalized)" font 'Helvetica,27'
set ylabel "Estimation Error" font 'Helvetica,27'
set output 'feature_scan/aws_good.eps'
set key font ",20"
plot "aws_feature_scan.csv" using 1:2:3 with errorbars lc rgb "blue" notitle,\
"aws_feature_scan.csv" using 1:2 with linespoints pt 7 lc rgb "blue" title "Fast video feature scanner(ours)" , \
"aws_feature_scan.csv" using 6:7:8 with errorbars lc rgb "red" notitle ,\
"aws_feature_scan.csv" using 6:7 with linespoints pt 7 lc rgb "red" title "Test VPS on randomly picked video segment" #font 'Helvetica,20'

reset
load 'style.gnu'
set datafile separator ","
set yrange [0:1]
set xrange [0:0.15]

set xlabel "GPU cycles(normalized)" font 'Helvetica,27'
set ylabel "Estimation Error" font 'Helvetica,27'
set output 'feature_scan/aws_bad.eps'
set key font ",20"
plot "aws_feature_scan.csv" using 1:4:5 with errorbars lc rgb "blue" notitle,\
"aws_feature_scan.csv" using 1:4 with linespoints pt 7 lc rgb "blue" title 'Fast video feature scanner(ours)', \
"aws_feature_scan.csv" using 6:9:10 with errorbars lc rgb "red" notitle,\
"aws_feature_scan.csv" using 6:9 with linespoints pt 7 lc rgb "red" title "Test VPS on randomly picked video segment"

reset
load 'style.gnu'
set datafile separator ","
set yrange [0:1]
set xrange [0:0.15]

set xlabel "GPU cycles(normalized)" font 'Helvetica,27'
set ylabel "Estimation Error" font 'Helvetica,27'
set output 'feature_scan/vs_good.eps'
set key font ",20"
plot "vs_feature_scan.csv" using 1:2:3 with errorbars lc rgb "blue" notitle,\
"vs_feature_scan.csv" using 1:2 with linespoints pt 7 lc rgb "blue" title "Fast video feature scanner(ours)" , \
"vs_feature_scan.csv" using 6:7:8 with errorbars lc rgb "red" notitle ,\
"vs_feature_scan.csv" using 6:7 with linespoints pt 7 lc rgb "red" title "Test VPS on randomly picked video segment" #font 'Helvetica,20'

reset
load 'style.gnu'
set datafile separator ","
set yrange [0:1]
set xrange [0:0.15]

set xlabel "GPU cycles(normalized)" font 'Helvetica,27'
set ylabel "Estimation Error" font 'Helvetica,27'
set output 'feature_scan/vs_bad.eps'
set key font ",20"
plot "vs_feature_scan.csv" using 1:4:5 with errorbars lc rgb "blue" notitle,\
"vs_feature_scan.csv" using 1:4 with linespoints pt 7 lc rgb "blue" title 'Fast video feature scanner(ours)', \
"vs_feature_scan.csv" using 6:9:10 with errorbars lc rgb "red" notitle,\
"vs_feature_scan.csv" using 6:9 with linespoints pt 7 lc rgb "red" title "Test VPS on randomly picked video segment"
