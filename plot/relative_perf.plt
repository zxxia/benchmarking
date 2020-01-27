#!/usr/bin/gnuplot -persist

reset
load 'style.gnu'
set datafile separator ","
# unset xtics
# unset ytics
set xrange [-0.2:0.2]
set yrange [-0.2:0.2]
set object 1 rectangle from -0.2,0.2 to 0,0 fc 'green' fillstyle solid 0.4 noborder
set object 2 rectangle from 0,0 to 0.2,-0.2 fc 'red' fillstyle solid 0.4 noborder
set xlabel "VideoStorm's cost relative to Glimpse's cost" font 'Helvetica,27'
set ylabel "VideoStorm's accuracy \nrelative to Glimpse's accuracy" font 'Helvetica,27'
set output 'relative_perf/gl_vs.eps'
set label "VideoStorm better\nthan Glimpse" at -0.18, 0.18 font 'Helvetica,25'
set label "VideoStorm worse\nthan Glimpse" at 0.02, -0.14 font 'Helvetica,25'
plot "gl_vs_improved.csv" using 1:2 pt 7 ps 1 lc rgb 'black' notitle


reset
load 'style.gnu'
set datafile separator ","
# unset xtics
# unset ytics
set xrange [-0.2:0.2]
set yrange [-0.2:0.2]
set object 1 rectangle from -0.2,0.2 to 0,0 fc 'green' fillstyle solid 0.4 noborder
set object 2 rectangle from 0,0 to 0.2,-0.2 fc 'red' fillstyle solid 0.4 noborder
set xlabel "Glimpse's cost relative to AWStream's cost" font 'Helvetica,27'
set ylabel "Glimpse's accuracy relative\nto AWStream's accuracy" font 'Helvetica,27'
set output 'relative_perf/aw_gl.eps'
set label "Glimpse better\nthan AWStream" at -0.18, 0.18 font 'Helvetica,25'
set label "Glimpse worse\nthan AWStream" at 0.02, -0.14 font 'Helvetica,25'
plot "aw_gl.csv" using 1:2 pt 7 ps 1 lc rgb 'black' notitle


# DEPRECATED!!!!
# reset
# load 'style.gnu'
# set datafile separator ","
# # unset xtics
# # unset ytics
# set xrange [-0.5:2.5]
# set yrange [-0.5:2.5]
# set object 1 rectangle from -0.5,2.5 to 1,1 fc 'green' fillstyle solid 0.4 noborder
# set object 2 rectangle from 1,1 to 2.5,-0.5 fc 'red' fillstyle solid 0.4 noborder
# set xlabel "Vigil's cost relative to AWStream's cost" font 'Helvetica,27'
# set ylabel "Vigil's accuracy relative\nto AWStream's accuracy" font 'Helvetica,27'
# set output 'relative_perf/aw_vg.eps'
# set label "Vigil strictly better\nthan AWStream" at -0.4, 2.2  font 'Helvetica,25'
# set label "Vigil strictly worse\n than AWStream" at 1.1, 0.0  font 'Helvetica,25'
# plot "aw_vg.csv" using 1:2 pt 7 ps 1 lc rgb 'black' notitle



# reset
# load 'style.gnu'
# set datafile separator ","
# # unset xtics
# # unset ytics
# set xrange [-0.5:2.5]
# set yrange [-0.5:2.5]
# set object 1 rectangle from -0.5,2.5 to 1,1 fc 'green' fillstyle solid 0.4 noborder
# set object 2 rectangle from 1,1 to 2.5,-0.5 fc 'red' fillstyle solid 0.4 noborder
# set xlabel "Chameleon's cost relative to NoScope's cost" font 'Helvetica,27'
# set ylabel "Chameleon's accuracy relative\nto NoScope's accuracy" font 'Helvetica,27'
# set output 'relative_perf/no_ch.eps'
# set label "Chameleon strictly\nbetter than NoScope" at -0.4, 2.2 font 'Helvetica,25'
# set label "Chameleon strictly\nworse than NoScope" at 1.1, 0.0 font 'Helvetica,25'
# plot "no_ch.csv" using 1:2 pt 7 ps 1 lc rgb 'black' notitle
