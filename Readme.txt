Steps to remember when marking:
1. Download map and search list and worlds/materials/textures/texture_ECE4078.jpeg
2. Rename map to... 7fruits_practice_map_1.txt
3. Place map in catkin_ws/src/penguinpi_gazebo using Ubuntu>home>charl>... manual copy and paste (may need to use dos2unix)
4. Place map in maps folder
5. Place search list in main folder
6. Place tecture_ECE4078.jpeg in catkin_ws/src/penguinpi_gazeb/worlds/materials/textures/texture_ECE4078.jpeg

7. Download code repo and place in Ubuntu>home>charl and rename as LiveDemo

8. source ~/LiveDemo/catkin_ws/devel/setup.bash
roslaunch penguinpi_gazebo ECE4078.launch

9. source ~/LiveDemo/catkin_ws/devel/setup.bash
rosrun penguinpi_gazebo scene_manager.py -l 7fruits_practice_map_1.txt

10. python3 slam_mapper.py
11. python3 TargetPoseEst.py 
12. python3 mapping_eval.py

13a. python3 AFR1.py
13b. python3 AFR2.py