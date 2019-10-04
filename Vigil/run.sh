#!/bin/bash


VIDEOS='traffic jp_hw russia tw_road tw_under_bridge nyc lane_split tw tw1
          russia1 park drift crossroad3 crossroad2 crossroad driving2
          crossroad4 driving1 driving_downtown highway highway_normal_traffic
          jp motorway'
VIDEOS='highway_normal_traffic'
for VIDEO in $VIDEOS
do
    python vigil_overfitting.py --video ${VIDEO}
done
