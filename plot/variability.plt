#!/usr/bin/gnuplot -persist

reset
load 'style.gnu'
set datafile separator ","
set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "Accuracy\n(Object Detection)" # font 'Helvetica,10'
set xlabel "Compute cost(GPU)" #font 'Helvetica,20'
# set label "Oracle\nBaseline" at 0.86, 0.93 font 'Helvetica,27'
set output 'var_across_videos/vs_var_across_videos.eps'
plot "vs_perf_acc.csv" using 1:2 pt 7 ps 0.6 lc rgb 'black' notitle,  \
     'vs_ellipse.csv' using 6:7:(0.01) with circles fillcolor 'white' fillstyle solid 1.0  border lc rgb 'red' lw 2 notitle
# 'baseline.csv' using 1:2 ls 2 ps 3.5 pt 9 lc rgb '#646464' notitle, \
# set object 2 circle at 0.1902,0.9091 radius char 0.15  fillstyle solid border lc rgb 'red' lw 2 front
# plot 'vs_ellipse.csv' using 1:2:3:4:5 with ellipse fc 'red' fillstyle solid 0.5 notitle,\
# set object 1 ellipse center 0.2121,0.8987 size 0.4579,0.2584 angle -168 fc 'red' fillstyle solid 0.5 #frontborder -1
# 'vs_ellipse.csv' using 6:7:(0.01) with circles notitle, \ #fillcolor 'white' solid 1.0  border lc rgb 'red' lw 2 notitle,\

reset
load 'style.gnu'
set datafile separator ","
# set term x11 1
set output 'var_across_videos/gl_var_across_videos.eps'
set xrange [0:1.1]
set yrange [0:1.1]
set ylabel "Accuracy\n(Object Detection)"# font 'Helvetica,10'
set xlabel "Compute cost(GPU)"# font 'Helvetica,10'
# set label "Oracle\nBaseline" at 0.86, 0.93 font 'Helvetica,27'
plot "gl_perf_acc.csv" using 1:2 pt 7 ps 0.6 lc rgb 'black' notitle, \
     'gl_ellipse.csv' using 6:7:(0.01) with circles fillcolor 'white' fillstyle solid 1.0  border lc rgb 'red' lw 2 notitle
# 'baseline.csv' using 1:2 ls 2 ps 3.5 pt 9 lc rgb '#646464' notitle
# plot 'gl_ellipse.csv' using 1:2:3:4:5 with ellipse fc 'red' fillstyle solid 0.5 notitle,\
# set obj 1 ellipse center 0.5702,0.8722 size 0.7552, 0.2611  angle -166 fc 'red' fillstyle solid 0.5

reset
load 'style.gnu'
set datafile separator ","
set output 'var_across_videos/aw_var_across_videos.eps'
set xrange [0:1.1]
set yrange [0:1.1]
# set obj 1 ellipse center 0.4599,0.8902 size 0.5944, 0.2484  angle -161 fc 'red' fillstyle solid 0.5
# set object 2 circle at 0.4561,0.9076 radius char 0.15  fillstyle solid border lc rgb 'red' lw 2 front

# set label "Oracle\nBaseline" at 0.86, 0.93 font 'Helvetica,27'
set ylabel "Accuracy\n(Object Detection)"
set xlabel "Network cost(Bandwidth)"
plot "aw_perf_acc.csv" using 1:2 pt 7 ps 0.6 lt 'black' notitle, \
     'aw_ellipse.csv' using 6:7:(0.01) with circles fillcolor 'white' fillstyle solid 1.0  border lc rgb 'red' lw 2 notitle
# 'baseline.csv' using 1:2 ls 2 ps 3.5 pt 9 lc rgb '#646464' notitle
# plot 'aw_ellipse.csv' using 1:2:3:4:5 with ellipse fc 'red' fillstyle solid 0.5 notitle,\



reset
load 'style.gnu'
set datafile separator ","
# set term x11 3
set output 'var_across_videos/no_var_across_videos.eps'
set xrange [0:1.1]
set yrange [0:1.1]
# set obj 1 ellipse center 0.1485,0.9544 size 0.8221, 0.2186  angle -174 fc 'red' fillstyle solid 0.5
# set object 2 circle at 0.0455,0.9766 radius char 0.15  fillstyle solid border lc rgb 'red' lw 2 front

# set label "Oracle\nBaseline" at 0.86, 0.93 font 'Helvetica,27'
set ylabel "Accuracy\n(Object Detection)"
set xlabel "Compute cost(GPU)"
plot "no_perf_acc.csv" using 1:2 pt 7 ps 0.6 lt 'black' notitle, \
     'no_ellipse.csv' using 6:7:(0.01) with circles fillcolor 'white' fillstyle solid 1.0  border lc rgb 'red' lw 2 notitle
# 'baseline.csv' using 1:2 ls 2 ps 3.5 pt 9 lc rgb '#646464' notitle
# plot 'no_ellipse.csv' using 1:2:3:4:5 with ellipse fc 'red' fillstyle solid 0.5 notitle,\
