#!/usr/bin/gnuplot -persist

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/vs_coverage_selected.eps'
set key right bottom
plot "vs_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "vs_video_selection.csv" using 3:4 pt 7 ps 0.8 lc rgb 'red' title 'Selected'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set key right bottom
set output 'video_selection/vs_coverage_kitti.eps'
plot "vs_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set',\
     "vs_video_selection.csv" using 5:6 pt 7 ps 0.8 lc rgb 'blue' title 'KITTI'
reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/vs_coverage_canada.eps'
plot "vs_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "vs_video_selection.csv" using 7:8 pt 7 ps 0.8 lc rgb 'green' title 'Long Video'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/aws_coverage_selected.eps'
plot "aws_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "aws_video_selection.csv" using 3:4 pt 7 ps 0.8 lc rgb 'red' title 'Selected'


reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/aws_coverage_kitti.eps'
plot "aws_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "aws_video_selection.csv" using 5:6 pt 7 ps 0.8 lc rgb 'blue' title 'KITTI'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/aws_coverage_crossroad2.eps'
plot "aws_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "aws_video_selection.csv" using 7:8 pt 7 ps 0.8 lc rgb 'green' title 'Long Video'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/no_coverage_selected.eps'
plot "no_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "no_video_selection.csv" using 3:4 pt 7 ps 0.8 lc rgb 'red' title 'Selected'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/no_coverage_kitti.eps'
plot "no_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "no_video_selection.csv" using 5:6 pt 7 ps 0.8 lc rgb 'blue' title 'KITTI'

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set key right bottom
set xlabel "Cost" font 'Helvetica,27'
set ylabel "Accuracy" font 'Helvetica,27'
set output 'video_selection/no_coverage_park.eps'
plot "no_video_selection.csv" using 1:2 pt 7 ps 0.8 lc rgb 'grey' title 'Coverage Set', \
     "no_video_selection.csv" using 7:8 pt 7 ps 0.8 lc rgb 'green' title 'Long Video'
