#!/bin/bash
declare -A RESOL
declare -A NUM_OF_FRAMES


# Change CODE_PATH to the object detection path.
CODE_PATH="/home/zxxia/models/research/object_detection"
# Change full model path to path where trained object detection model is saved.
FULL_MODEL_PATH="/home/zxxia/models/research/"\
"object_detection/faster_rcnn_resnet101_coco_2018_01_28/"\
"frozen_inference_graph.pb"

# Change DATA_PATH to downloaded Youtube video path
# DATA_PATH='/mnt/data/zhujun/dataset/Youtube/'
DATA_PATH='/mnt/data/zhujun/new_video/validation_0000/'
# Change dataset_list to your dataset name
DATASET_LIST="segment-10448102132863604198_472_000_492_000
segment-10689101165701914459_2072_300_2092_300
segment-11037651371539287009_77_670_97_670
segment-1105338229944737854_1280_000_1300_000
segment-11660186733224028707_420_000_440_000
segment-13178092897340078601_5118_604_5138_604
segment-14127943473592757944_2068_000_2088_000
segment-14956919859981065721_1759_980_1779_980
segment-16213317953898915772_1597_170_1617_170
segment-16751706457322889693_4475_240_4495_240
segment-17344036177686610008_7852_160_7872_160
segment-17539775446039009812_440_000_460_000
segment-17612470202990834368_2800_000_2820_000
segment-2094681306939952000_2972_300_2992_300
segment-272435602399417322_2884_130_2904_130
segment-4423389401016162461_4235_900_4255_900
segment-4816728784073043251_5273_410_5293_410
segment-6001094526418694294_4609_470_4629_470
segment-6074871217133456543_1000_000_1020_000
segment-6324079979569135086_2372_300_2392_300
segment-662188686397364823_3248_800_3268_800
segment-6637600600814023975_2235_000_2255_000
segment-8845277173853189216_3828_530_3848_530
segment-8888517708810165484_1549_770_1569_770
segment-902001779062034993_2880_000_2900_000
"
#"segment-4458730539804900192_535_000_555_000/
#segment-4604173119409817302_2820_000_2840_000/
#segment-4672649953433758614_2700_000_2720_000/
#segment-4781039348168995891_280_000_300_000/
#segment-5100136784230856773_2517_300_2537_300/
#segment-5127440443725457056_2921_340_2941_340/
#segment-5200186706748209867_80_000_100_000/
#segment-5214491533551928383_1918_780_1938_780/
#segment-54293441958058219_2335_200_2355_200/
#segment-5446766520699850364_157_000_177_000/
#segment-5525943706123287091_4100_000_4120_000/
#segment-5526948896847934178_1039_000_1059_000/
#segment-6234738900256277070_320_000_340_000/
#segment-6350707596465488265_2393_900_2413_900/
#segment-6694593639447385226_1040_000_1060_000/
#segment-6771922013310347577_4249_290_4269_290/
#segment-7458568461947999548_700_000_720_000/
#segment-7670103006580549715_360_000_380_000/
#segment-7741361323303179462_1230_310_1250_310/
#segment-7890808800227629086_6162_700_6182_700/
#segment-8031709558315183746_491_220_511_220/
#segment-8513241054672631743_115_960_135_960/
#segment-8700094808505895018_7272_488_7292_488/
#segment-9250355398701464051_4166_132_4186_132/
#segment-9320169289978396279_1040_000_1060_000/
#"


# "segment-15857303257471811288_1840_000_1860_000
# segment-15868625208244306149_4340_000_4360_000
# segment-16102220208346880_1420_000_1440_000
# segment-16341778301681295961_178_800_198_800
# segment-16652690380969095006_2580_000_2600_000
# segment-17597174721305220109_178_000_198_000
# segment-18111897798871103675_320_000_340_000
# segment-18244334282518155052_2360_000_2380_000
# segment-18380281348728758158_4820_000_4840_000
# segment-200287570390499785_2102_000_2122_000
# segment-2064489349728221803_3060_000_3080_000
# segment-2607999228439188545_2960_000_2980_000
# segment-2739239662326039445_5890_320_5910_320
# segment-2961247865039433386_920_000_940_000
# segment-3002379261592154728_2256_691_2276_691
# segment-3270384983482134275_3220_000_3240_000
# segment-3276301746183196185_436_450_456_450
# segment-33101359476901423_6720_910_6740_910
# segment-3375636961848927657_1942_000_1962_000
# segment-3417928259332148981_7018_550_7038_550
# segment-3441838785578020259_1300_000_1320_000
# segment-384975055665199088_4480_000_4500_000
# segment-3908622028474148527_3480_000_3500_000
# segment-4058410353286511411_3980_000_4000_000
# "

# "segment-10206293520369375008_2796_800_2816_800
# segment-10241508783381919015_2889_360_2909_360
# segment-10500357041547037089_1474_800_1494_800
# segment-10526338824408452410_5714_660_5734_660
# segment-10724020115992582208_7660_400_7680_400
# segment-11004685739714500220_2300_000_2320_000
# segment-11119453952284076633_1369_940_1389_940
# segment-11355519273066561009_5323_000_5343_000
# segment-11623618970700582562_2840_367_2860_367
# segment-11799592541704458019_9828_750_9848_750
# segment-1208303279778032257_1360_000_1380_000
# segment-12257951615341726923_2196_690_2216_690
# segment-12304907743194762419_1522_000_1542_000
# segment-12581809607914381746_1219_547_1239_547
# segment-13519445614718437933_4060_000_4080_000
# segment-14106113060128637865_1200_000_1220_000
# segment-14143054494855609923_4529_100_4549_100
# segment-14233522945839943589_100_000_120_000
# segment-14753089714893635383_873_600_893_600
# segment-15036582848618865396_3752_830_3772_830
# segment-15374821596407640257_3388_480_3408_480
# segment-15445436653637630344_3957_561_3977_561
# segment-15533468984793020049_800_000_820_000
# segment-15578655130939579324_620_000_640_000"

#DATASET_LIST="highway_normal_traffic russia1 drift"
#"08041754 08041847 08041951 08042133 08050938 08051137"
#"akiyo_cif deadline_cif ice_4cif pedestrian_area_1080p25 "\
#"bowing_cif football_422_ntsc KristenAndSara_1280x720_60 "\
#"rush_hour_1080p25 bus_cif foreman_cif mad900_cif "\
#"station2_1080p25 carphone_qcif FourPeople_1280x720_60 "\
#"miss_am_qcif tractor_1080p25 claire_qcif grandma_qcif "\
#"mthr_dotr_qcif coastguard_cif hall_objects_qcif "\
#"Netflix_Crosswalk_4096x2160_60fps_10bit_420 crew_4cif "\
#"highway_cif Netflix_DrivingPOV_4096x2160_60fps_10bit_420"
#"jp highway motorway" # driving_downtown crossroad4 crossroad2 crossroad driving1 russia russia1"
#"jp russia1 tw tw1 park"
# Choose an idle GPU
GPU="2"
RESIZE_RESOL_LIST='720p 540p 480p 360p'
QP_LIST="original" #'30 35 40'
for DATASET_NAME in $DATASET_LIST
do
    echo ${DATASET_NAME}

    for RESIZE_RESOL in $RESIZE_RESOL_LIST
    do
        #echo $RESIZE_RESOL
        #python3 resize.py \
            #--dataset=${DATASET_NAME} \
            #--resize_resol=$RESIZE_RESOL \
            #--path=$DATA_PATH

        for QP in $QP_LIST
        do
            #python3 change_quantization.py \
                #--dataset=$DATASET_NAME \
                #--path=$DATA_PATH \
                #--quality_parameter=$QP \
                #--resolution=$RESIZE_RESOL

            python3 ${CODE_PATH}/dataset_tools/create_youtube_video_input.py \
                --dataset=${DATASET_NAME} \
                --path=$DATA_PATH \
                --resize_resol=$RESIZE_RESOL \
                --quality_parameter=$QP

            echo "Done creating input!"
            python3 ${CODE_PATH}/inference/infer_detections_for_ground_truth.py \
                --inference_graph=$FULL_MODEL_PATH \
                --discard_image_pixels \
                --dataset=${DATASET_NAME} \
                --gpu=$GPU \
                --path=$DATA_PATH \
                --resize_resol=$RESIZE_RESOL \
                --quality_parameter=$QP

            python3 infer_object_id_youtube.py \
                --dataset=${DATASET_NAME} \
                --path=$DATA_PATH \
                --resize_resol=$RESIZE_RESOL \
                --quality_parameter=$QP

            if [ "$RESIZE_RESOL" = original ]; then
                python3 video_feature_youtube.py \
                    --dataset=${DATASET_NAME} \
                    --path=$DATA_PATH
            fi
        done
    done

done
