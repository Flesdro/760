# Focal
```
==================================================
Training: Baseline (Local: BoW)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0097 | Val mAP: 0.1158 | Mi-F1: 0.1646 | Ma-F1: 0.0139
Epoch [02/35] Loss: 0.0083 | Val mAP: 0.1319 | Mi-F1: 0.1989 | Ma-F1: 0.0196
Epoch [03/35] Loss: 0.0081 | Val mAP: 0.1380 | Mi-F1: 0.1815 | Ma-F1: 0.0195
Epoch [04/35] Loss: 0.0078 | Val mAP: 0.1440 | Mi-F1: 0.2224 | Ma-F1: 0.0271
Epoch [05/35] Loss: 0.0077 | Val mAP: 0.1473 | Mi-F1: 0.2338 | Ma-F1: 0.0316
Epoch [06/35] Loss: 0.0079 | Val mAP: 0.1501 | Mi-F1: 0.2418 | Ma-F1: 0.0321
Epoch [07/35] Loss: 0.0075 | Val mAP: 0.1505 | Mi-F1: 0.2379 | Ma-F1: 0.0319
Epoch [08/35] Loss: 0.0074 | Val mAP: 0.1513 | Mi-F1: 0.2333 | Ma-F1: 0.0344
Epoch [09/35] Loss: 0.0076 | Val mAP: 0.1528 | Mi-F1: 0.2554 | Ma-F1: 0.0374
Epoch [10/35] Loss: 0.0072 | Val mAP: 0.1522 | Mi-F1: 0.2724 | Ma-F1: 0.0399
Epoch [11/35] Loss: 0.0071 | Val mAP: 0.1522 | Mi-F1: 0.2778 | Ma-F1: 0.0417
Epoch [12/35] Loss: 0.0070 | Val mAP: 0.1508 | Mi-F1: 0.2528 | Ma-F1: 0.0394
Epoch [13/35] Loss: 0.0069 | Val mAP: 0.1508 | Mi-F1: 0.2609 | Ma-F1: 0.0445
Epoch [14/35] Loss: 0.0068 | Val mAP: 0.1493 | Mi-F1: 0.2834 | Ma-F1: 0.0438
Epoch [15/35] Loss: 0.0067 | Val mAP: 0.1493 | Mi-F1: 0.2711 | Ma-F1: 0.0439
Epoch [16/35] Loss: 0.0066 | Val mAP: 0.1486 | Mi-F1: 0.2879 | Ma-F1: 0.0450
Epoch [17/35] Loss: 0.0065 | Val mAP: 0.1478 | Mi-F1: 0.2875 | Ma-F1: 0.0471
Epoch [18/35] Loss: 0.0066 | Val mAP: 0.1468 | Mi-F1: 0.3043 | Ma-F1: 0.0527
Epoch [19/35] Loss: 0.0064 | Val mAP: 0.1456 | Mi-F1: 0.2862 | Ma-F1: 0.0491
Epoch [20/35] Loss: 0.0063 | Val mAP: 0.1454 | Mi-F1: 0.2929 | Ma-F1: 0.0500
Epoch [21/35] Loss: 0.0063 | Val mAP: 0.1434 | Mi-F1: 0.2983 | Ma-F1: 0.0501
Epoch [22/35] Loss: 0.0062 | Val mAP: 0.1410 | Mi-F1: 0.2911 | Ma-F1: 0.0499
Epoch [23/35] Loss: 0.0062 | Val mAP: 0.1420 | Mi-F1: 0.3027 | Ma-F1: 0.0512
Epoch [24/35] Loss: 0.0061 | Val mAP: 0.1424 | Mi-F1: 0.3090 | Ma-F1: 0.0534
Epoch [25/35] Loss: 0.0062 | Val mAP: 0.1408 | Mi-F1: 0.3113 | Ma-F1: 0.0541
Epoch [26/35] Loss: 0.0060 | Val mAP: 0.1411 | Mi-F1: 0.3081 | Ma-F1: 0.0540
Epoch [27/35] Loss: 0.0060 | Val mAP: 0.1405 | Mi-F1: 0.3094 | Ma-F1: 0.0545
Epoch [28/35] Loss: 0.0059 | Val mAP: 0.1388 | Mi-F1: 0.3072 | Ma-F1: 0.0536
Epoch [29/35] Loss: 0.0059 | Val mAP: 0.1391 | Mi-F1: 0.3237 | Ma-F1: 0.0592
Epoch [30/35] Loss: 0.0059 | Val mAP: 0.1379 | Mi-F1: 0.3065 | Ma-F1: 0.0539
Epoch [31/35] Loss: 0.0058 | Val mAP: 0.1378 | Mi-F1: 0.3184 | Ma-F1: 0.0566
Epoch [32/35] Loss: 0.0058 | Val mAP: 0.1363 | Mi-F1: 0.3151 | Ma-F1: 0.0532
Epoch [33/35] Loss: 0.0058 | Val mAP: 0.1372 | Mi-F1: 0.3152 | Ma-F1: 0.0560
Epoch [34/35] Loss: 0.0057 | Val mAP: 0.1358 | Mi-F1: 0.3196 | Ma-F1: 0.0551
Epoch [35/35] Loss: 0.0057 | Val mAP: 0.1354 | Mi-F1: 0.3141 | Ma-F1: 0.0551
[TEST SET] mAP: 0.1737 | Micro-F1: 0.2873 | Macro-F1: 0.0407

==================================================
Training: Baseline (Global: Color+Texture)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0098 | Val mAP: 0.1846 | Mi-F1: 0.2940 | Ma-F1: 0.0381
Epoch [02/35] Loss: 0.0070 | Val mAP: 0.2194 | Mi-F1: 0.3485 | Ma-F1: 0.0497
Epoch [03/35] Loss: 0.0068 | Val mAP: 0.2420 | Mi-F1: 0.3691 | Ma-F1: 0.0542
Epoch [04/35] Loss: 0.0066 | Val mAP: 0.2556 | Mi-F1: 0.3780 | Ma-F1: 0.0642
Epoch [05/35] Loss: 0.0065 | Val mAP: 0.2641 | Mi-F1: 0.3923 | Ma-F1: 0.0709
Epoch [06/35] Loss: 0.0064 | Val mAP: 0.2740 | Mi-F1: 0.4000 | Ma-F1: 0.0739
Epoch [07/35] Loss: 0.0063 | Val mAP: 0.2816 | Mi-F1: 0.4002 | Ma-F1: 0.0790
Epoch [08/35] Loss: 0.0063 | Val mAP: 0.2860 | Mi-F1: 0.4052 | Ma-F1: 0.0858
Epoch [09/35] Loss: 0.0062 | Val mAP: 0.2907 | Mi-F1: 0.4168 | Ma-F1: 0.0873
Epoch [10/35] Loss: 0.0062 | Val mAP: 0.2941 | Mi-F1: 0.4296 | Ma-F1: 0.0939
Epoch [11/35] Loss: 0.0061 | Val mAP: 0.2976 | Mi-F1: 0.4221 | Ma-F1: 0.0903
Epoch [12/35] Loss: 0.0061 | Val mAP: 0.3022 | Mi-F1: 0.4249 | Ma-F1: 0.0927
Epoch [13/35] Loss: 0.0061 | Val mAP: 0.3039 | Mi-F1: 0.4372 | Ma-F1: 0.1018
Epoch [14/35] Loss: 0.0060 | Val mAP: 0.3053 | Mi-F1: 0.4336 | Ma-F1: 0.1071
Epoch [15/35] Loss: 0.0060 | Val mAP: 0.3075 | Mi-F1: 0.4415 | Ma-F1: 0.1077
Epoch [16/35] Loss: 0.0059 | Val mAP: 0.3108 | Mi-F1: 0.4544 | Ma-F1: 0.1141
Epoch [17/35] Loss: 0.0059 | Val mAP: 0.3127 | Mi-F1: 0.4482 | Ma-F1: 0.1132
Epoch [18/35] Loss: 0.0059 | Val mAP: 0.3134 | Mi-F1: 0.4476 | Ma-F1: 0.1184
Epoch [19/35] Loss: 0.0058 | Val mAP: 0.3146 | Mi-F1: 0.4601 | Ma-F1: 0.1209
Epoch [20/35] Loss: 0.0058 | Val mAP: 0.3166 | Mi-F1: 0.4582 | Ma-F1: 0.1264
Epoch [21/35] Loss: 0.0058 | Val mAP: 0.3179 | Mi-F1: 0.4650 | Ma-F1: 0.1243
Epoch [22/35] Loss: 0.0058 | Val mAP: 0.3179 | Mi-F1: 0.4503 | Ma-F1: 0.1207
Epoch [23/35] Loss: 0.0057 | Val mAP: 0.3200 | Mi-F1: 0.4592 | Ma-F1: 0.1328
Epoch [24/35] Loss: 0.0057 | Val mAP: 0.3198 | Mi-F1: 0.4641 | Ma-F1: 0.1345
Epoch [25/35] Loss: 0.0057 | Val mAP: 0.3204 | Mi-F1: 0.4668 | Ma-F1: 0.1375
Epoch [26/35] Loss: 0.0057 | Val mAP: 0.3213 | Mi-F1: 0.4598 | Ma-F1: 0.1342
Epoch [27/35] Loss: 0.0056 | Val mAP: 0.3208 | Mi-F1: 0.4679 | Ma-F1: 0.1367
Epoch [28/35] Loss: 0.0056 | Val mAP: 0.3239 | Mi-F1: 0.4693 | Ma-F1: 0.1404
Epoch [29/35] Loss: 0.0056 | Val mAP: 0.3229 | Mi-F1: 0.4710 | Ma-F1: 0.1402
Epoch [30/35] Loss: 0.0056 | Val mAP: 0.3245 | Mi-F1: 0.4733 | Ma-F1: 0.1413
Epoch [31/35] Loss: 0.0055 | Val mAP: 0.3257 | Mi-F1: 0.4814 | Ma-F1: 0.1460
Epoch [32/35] Loss: 0.0055 | Val mAP: 0.3254 | Mi-F1: 0.4772 | Ma-F1: 0.1440
Epoch [33/35] Loss: 0.0055 | Val mAP: 0.3250 | Mi-F1: 0.4888 | Ma-F1: 0.1489
Epoch [34/35] Loss: 0.0055 | Val mAP: 0.3265 | Mi-F1: 0.4783 | Ma-F1: 0.1473
Epoch [35/35] Loss: 0.0055 | Val mAP: 0.3261 | Mi-F1: 0.4781 | Ma-F1: 0.1465
[TEST SET] mAP: 0.3444 | Micro-F1: 0.4571 | Macro-F1: 0.1407

==================================================
Training: Early Fusion (Concat)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0088 | Val mAP: 0.1864 | Mi-F1: 0.3570 | Ma-F1: 0.0479
Epoch [02/35] Loss: 0.0070 | Val mAP: 0.2224 | Mi-F1: 0.3668 | Ma-F1: 0.0545
Epoch [03/35] Loss: 0.0066 | Val mAP: 0.2427 | Mi-F1: 0.4069 | Ma-F1: 0.0674
Epoch [04/35] Loss: 0.0064 | Val mAP: 0.2568 | Mi-F1: 0.4321 | Ma-F1: 0.0760
Epoch [05/35] Loss: 0.0063 | Val mAP: 0.2693 | Mi-F1: 0.4292 | Ma-F1: 0.0847
Epoch [06/35] Loss: 0.0062 | Val mAP: 0.2765 | Mi-F1: 0.4204 | Ma-F1: 0.0882
Epoch [07/35] Loss: 0.0061 | Val mAP: 0.2820 | Mi-F1: 0.4367 | Ma-F1: 0.0916
Epoch [08/35] Loss: 0.0060 | Val mAP: 0.2873 | Mi-F1: 0.4475 | Ma-F1: 0.1040
Epoch [09/35] Loss: 0.0059 | Val mAP: 0.2920 | Mi-F1: 0.4602 | Ma-F1: 0.1109
Epoch [10/35] Loss: 0.0058 | Val mAP: 0.2950 | Mi-F1: 0.4520 | Ma-F1: 0.1110
Epoch [11/35] Loss: 0.0058 | Val mAP: 0.2971 | Mi-F1: 0.4676 | Ma-F1: 0.1186
Epoch [12/35] Loss: 0.0057 | Val mAP: 0.3003 | Mi-F1: 0.4581 | Ma-F1: 0.1168
Epoch [13/35] Loss: 0.0056 | Val mAP: 0.3001 | Mi-F1: 0.4700 | Ma-F1: 0.1177
Epoch [14/35] Loss: 0.0056 | Val mAP: 0.3033 | Mi-F1: 0.4779 | Ma-F1: 0.1284
Epoch [15/35] Loss: 0.0055 | Val mAP: 0.3053 | Mi-F1: 0.4720 | Ma-F1: 0.1264
Epoch [16/35] Loss: 0.0055 | Val mAP: 0.3049 | Mi-F1: 0.4711 | Ma-F1: 0.1321
Epoch [17/35] Loss: 0.0054 | Val mAP: 0.3059 | Mi-F1: 0.4748 | Ma-F1: 0.1378
Epoch [18/35] Loss: 0.0054 | Val mAP: 0.3052 | Mi-F1: 0.4901 | Ma-F1: 0.1308
Epoch [19/35] Loss: 0.0053 | Val mAP: 0.3062 | Mi-F1: 0.4875 | Ma-F1: 0.1373
Epoch [20/35] Loss: 0.0053 | Val mAP: 0.3077 | Mi-F1: 0.4814 | Ma-F1: 0.1388
Epoch [21/35] Loss: 0.0052 | Val mAP: 0.3071 | Mi-F1: 0.4950 | Ma-F1: 0.1442
Epoch [22/35] Loss: 0.0052 | Val mAP: 0.3076 | Mi-F1: 0.4888 | Ma-F1: 0.1479
Epoch [23/35] Loss: 0.0051 | Val mAP: 0.3072 | Mi-F1: 0.4929 | Ma-F1: 0.1568
Epoch [24/35] Loss: 0.0051 | Val mAP: 0.3068 | Mi-F1: 0.4924 | Ma-F1: 0.1510
Epoch [25/35] Loss: 0.0051 | Val mAP: 0.3065 | Mi-F1: 0.4932 | Ma-F1: 0.1514
Epoch [26/35] Loss: 0.0050 | Val mAP: 0.3069 | Mi-F1: 0.4936 | Ma-F1: 0.1549
Epoch [27/35] Loss: 0.0050 | Val mAP: 0.3054 | Mi-F1: 0.5068 | Ma-F1: 0.1557
Epoch [28/35] Loss: 0.0049 | Val mAP: 0.3058 | Mi-F1: 0.4906 | Ma-F1: 0.1596
Epoch [29/35] Loss: 0.0049 | Val mAP: 0.3065 | Mi-F1: 0.5030 | Ma-F1: 0.1679
Epoch [30/35] Loss: 0.0049 | Val mAP: 0.3055 | Mi-F1: 0.4970 | Ma-F1: 0.1604
Epoch [31/35] Loss: 0.0049 | Val mAP: 0.3063 | Mi-F1: 0.5039 | Ma-F1: 0.1680
Epoch [32/35] Loss: 0.0048 | Val mAP: 0.3062 | Mi-F1: 0.5082 | Ma-F1: 0.1708
Epoch [33/35] Loss: 0.0048 | Val mAP: 0.3044 | Mi-F1: 0.5056 | Ma-F1: 0.1668
Epoch [34/35] Loss: 0.0048 | Val mAP: 0.3038 | Mi-F1: 0.5092 | Ma-F1: 0.1713
Epoch [35/35] Loss: 0.0047 | Val mAP: 0.3034 | Mi-F1: 0.5140 | Ma-F1: 0.1682
[TEST SET] mAP: 0.3667 | Micro-F1: 0.4907 | Macro-F1: 0.1578

==================================================
Training: Gated Fusion + InfoNCE
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.1724 | Val mAP: 0.1215 | Mi-F1: 0.2106 | Ma-F1: 0.0171
Epoch [02/35] Loss: 0.1220 | Val mAP: 0.1512 | Mi-F1: 0.2584 | Ma-F1: 0.0254
Epoch [03/35] Loss: 0.1063 | Val mAP: 0.1719 | Mi-F1: 0.3080 | Ma-F1: 0.0355
Epoch [04/35] Loss: 0.0970 | Val mAP: 0.1882 | Mi-F1: 0.3131 | Ma-F1: 0.0395
Epoch [05/35] Loss: 0.0910 | Val mAP: 0.2019 | Mi-F1: 0.3159 | Ma-F1: 0.0424
Epoch [06/35] Loss: 0.0872 | Val mAP: 0.2104 | Mi-F1: 0.3408 | Ma-F1: 0.0467
Epoch [07/35] Loss: 0.0842 | Val mAP: 0.2187 | Mi-F1: 0.3673 | Ma-F1: 0.0500
Epoch [08/35] Loss: 0.0821 | Val mAP: 0.2255 | Mi-F1: 0.3773 | Ma-F1: 0.0552
Epoch [09/35] Loss: 0.0806 | Val mAP: 0.2311 | Mi-F1: 0.3777 | Ma-F1: 0.0569
Epoch [10/35] Loss: 0.0793 | Val mAP: 0.2355 | Mi-F1: 0.3690 | Ma-F1: 0.0570
Epoch [11/35] Loss: 0.0781 | Val mAP: 0.2400 | Mi-F1: 0.3733 | Ma-F1: 0.0585
Epoch [12/35] Loss: 0.0770 | Val mAP: 0.2420 | Mi-F1: 0.3887 | Ma-F1: 0.0679
Epoch [13/35] Loss: 0.0763 | Val mAP: 0.2452 | Mi-F1: 0.3866 | Ma-F1: 0.0632
Epoch [14/35] Loss: 0.0756 | Val mAP: 0.2487 | Mi-F1: 0.3853 | Ma-F1: 0.0695
Epoch [15/35] Loss: 0.0746 | Val mAP: 0.2493 | Mi-F1: 0.3917 | Ma-F1: 0.0675
Epoch [16/35] Loss: 0.0741 | Val mAP: 0.2527 | Mi-F1: 0.3961 | Ma-F1: 0.0728
Epoch [17/35] Loss: 0.0734 | Val mAP: 0.2551 | Mi-F1: 0.3887 | Ma-F1: 0.0732
Epoch [18/35] Loss: 0.0728 | Val mAP: 0.2568 | Mi-F1: 0.4044 | Ma-F1: 0.0811
Epoch [19/35] Loss: 0.0724 | Val mAP: 0.2586 | Mi-F1: 0.4025 | Ma-F1: 0.0751
Epoch [20/35] Loss: 0.0719 | Val mAP: 0.2592 | Mi-F1: 0.4173 | Ma-F1: 0.0787
Epoch [21/35] Loss: 0.0714 | Val mAP: 0.2613 | Mi-F1: 0.3983 | Ma-F1: 0.0802
Epoch [22/35] Loss: 0.0711 | Val mAP: 0.2634 | Mi-F1: 0.4075 | Ma-F1: 0.0783
Epoch [23/35] Loss: 0.0707 | Val mAP: 0.2631 | Mi-F1: 0.4044 | Ma-F1: 0.0784
Epoch [24/35] Loss: 0.0704 | Val mAP: 0.2646 | Mi-F1: 0.4163 | Ma-F1: 0.0827
Epoch [25/35] Loss: 0.0700 | Val mAP: 0.2668 | Mi-F1: 0.4125 | Ma-F1: 0.0845
Epoch [26/35] Loss: 0.0697 | Val mAP: 0.2678 | Mi-F1: 0.4169 | Ma-F1: 0.0845
Epoch [27/35] Loss: 0.0694 | Val mAP: 0.2692 | Mi-F1: 0.4178 | Ma-F1: 0.0841
Epoch [28/35] Loss: 0.0690 | Val mAP: 0.2699 | Mi-F1: 0.4185 | Ma-F1: 0.0879
Epoch [29/35] Loss: 0.0689 | Val mAP: 0.2697 | Mi-F1: 0.4100 | Ma-F1: 0.0880
Epoch [30/35] Loss: 0.0687 | Val mAP: 0.2707 | Mi-F1: 0.4312 | Ma-F1: 0.0908
Epoch [31/35] Loss: 0.0685 | Val mAP: 0.2732 | Mi-F1: 0.4209 | Ma-F1: 0.0919
Epoch [32/35] Loss: 0.0682 | Val mAP: 0.2736 | Mi-F1: 0.4197 | Ma-F1: 0.0937
Epoch [33/35] Loss: 0.0680 | Val mAP: 0.2726 | Mi-F1: 0.4166 | Ma-F1: 0.0907
Epoch [34/35] Loss: 0.0678 | Val mAP: 0.2721 | Mi-F1: 0.4285 | Ma-F1: 0.0909
Epoch [35/35] Loss: 0.0676 | Val mAP: 0.2762 | Mi-F1: 0.4220 | Ma-F1: 0.0919
[TEST SET] mAP: 0.3172 | Micro-F1: 0.3898 | Macro-F1: 0.0735

==================================================
Training: Gated Fusion (No InfoNCE)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0090 | Val mAP: 0.1750 | Mi-F1: 0.3457 | Ma-F1: 0.0455
Epoch [02/35] Loss: 0.0069 | Val mAP: 0.2188 | Mi-F1: 0.3667 | Ma-F1: 0.0514
Epoch [03/35] Loss: 0.0066 | Val mAP: 0.2404 | Mi-F1: 0.3992 | Ma-F1: 0.0622
Epoch [04/35] Loss: 0.0064 | Val mAP: 0.2548 | Mi-F1: 0.4230 | Ma-F1: 0.0715
Epoch [05/35] Loss: 0.0063 | Val mAP: 0.2649 | Mi-F1: 0.4314 | Ma-F1: 0.0841
Epoch [06/35] Loss: 0.0062 | Val mAP: 0.2732 | Mi-F1: 0.4408 | Ma-F1: 0.0948
Epoch [07/35] Loss: 0.0061 | Val mAP: 0.2811 | Mi-F1: 0.4384 | Ma-F1: 0.0939
Epoch [08/35] Loss: 0.0060 | Val mAP: 0.2851 | Mi-F1: 0.4552 | Ma-F1: 0.1057
Epoch [09/35] Loss: 0.0059 | Val mAP: 0.2876 | Mi-F1: 0.4462 | Ma-F1: 0.0993
Epoch [10/35] Loss: 0.0059 | Val mAP: 0.2914 | Mi-F1: 0.4457 | Ma-F1: 0.1052
Epoch [11/35] Loss: 0.0058 | Val mAP: 0.2933 | Mi-F1: 0.4483 | Ma-F1: 0.1048
Epoch [12/35] Loss: 0.0058 | Val mAP: 0.2960 | Mi-F1: 0.4719 | Ma-F1: 0.1120
Epoch [13/35] Loss: 0.0057 | Val mAP: 0.2976 | Mi-F1: 0.4547 | Ma-F1: 0.1159
Epoch [14/35] Loss: 0.0056 | Val mAP: 0.2973 | Mi-F1: 0.4682 | Ma-F1: 0.1150
Epoch [15/35] Loss: 0.0056 | Val mAP: 0.3005 | Mi-F1: 0.4727 | Ma-F1: 0.1211
Epoch [16/35] Loss: 0.0055 | Val mAP: 0.2996 | Mi-F1: 0.4767 | Ma-F1: 0.1258
Epoch [17/35] Loss: 0.0055 | Val mAP: 0.3004 | Mi-F1: 0.4734 | Ma-F1: 0.1282
Epoch [18/35] Loss: 0.0055 | Val mAP: 0.3043 | Mi-F1: 0.4812 | Ma-F1: 0.1379
Epoch [19/35] Loss: 0.0054 | Val mAP: 0.3040 | Mi-F1: 0.4694 | Ma-F1: 0.1323
Epoch [20/35] Loss: 0.0056 | Val mAP: 0.3034 | Mi-F1: 0.4801 | Ma-F1: 0.1342
Epoch [21/35] Loss: 0.0053 | Val mAP: 0.3013 | Mi-F1: 0.4904 | Ma-F1: 0.1430
Epoch [22/35] Loss: 0.0053 | Val mAP: 0.3036 | Mi-F1: 0.4821 | Ma-F1: 0.1472
Epoch [23/35] Loss: 0.0053 | Val mAP: 0.3038 | Mi-F1: 0.4763 | Ma-F1: 0.1463
Epoch [24/35] Loss: 0.0052 | Val mAP: 0.3017 | Mi-F1: 0.4940 | Ma-F1: 0.1494
Epoch [25/35] Loss: 0.0052 | Val mAP: 0.3039 | Mi-F1: 0.4964 | Ma-F1: 0.1580
Epoch [26/35] Loss: 0.0052 | Val mAP: 0.3023 | Mi-F1: 0.4917 | Ma-F1: 0.1502
Epoch [27/35] Loss: 0.0051 | Val mAP: 0.3024 | Mi-F1: 0.4959 | Ma-F1: 0.1563
Epoch [28/35] Loss: 0.0051 | Val mAP: 0.3016 | Mi-F1: 0.4993 | Ma-F1: 0.1527
Epoch [29/35] Loss: 0.0051 | Val mAP: 0.3025 | Mi-F1: 0.4987 | Ma-F1: 0.1585
Epoch [30/35] Loss: 0.0050 | Val mAP: 0.3008 | Mi-F1: 0.4973 | Ma-F1: 0.1566
Epoch [31/35] Loss: 0.0051 | Val mAP: 0.3004 | Mi-F1: 0.4930 | Ma-F1: 0.1668
Epoch [32/35] Loss: 0.0052 | Val mAP: 0.2987 | Mi-F1: 0.4920 | Ma-F1: 0.1630
Epoch [33/35] Loss: 0.0049 | Val mAP: 0.3004 | Mi-F1: 0.5020 | Ma-F1: 0.1622
Epoch [34/35] Loss: 0.0049 | Val mAP: 0.3001 | Mi-F1: 0.4981 | Ma-F1: 0.1654
Epoch [35/35] Loss: 0.0049 | Val mAP: 0.2986 | Mi-F1: 0.4908 | Ma-F1: 0.1700
[TEST SET] mAP: 0.3566 | Micro-F1: 0.4623 | Macro-F1: 0.1653

====================================================================
Experiment Model               | mAP (%)  | Micro-F1 | Macro-F1
--------------------------------------------------------------------
Baseline_A (BoW)               | 17.37     | 28.73     | 4.07
Baseline_B (Color+Tex)         | 34.44     | 45.71     | 14.07
Early_Fusion (Concat)          | 36.67     | 49.07     | 15.78
Gated_Fusion + InfoNCE         | 31.72     | 38.98     | 7.35
Gated_Fusion (No NCE)          | 35.66     | 46.23     | 16.53
====================================================================
```

## Analysis

The Focal Loss experiments show that **Early Fusion (Concat)** achieves the best mAP, while **Gated Fusion (No InfoNCE)** achieves the best Macro-F1:

| Model | mAP (%) | Micro-F1 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Baseline_A (BoW) | 17.37 | 28.73 | 4.07 |
| Baseline_B (Color+Texture) | 34.44 | 45.71 | 14.07 |
| Early_Fusion (Concat) | **36.67** | **49.07** | 15.78 |
| Gated_Fusion + InfoNCE | 31.72 | 38.98 | 7.35 |
| Gated_Fusion (No InfoNCE) | 35.66 | 46.23 | **16.53** |

### Main Findings

First, the same feature trend appears again: **Color+Texture** is much stronger than **BoW**. The BoW-only model reaches only 17.37% mAP and 4.07% Macro-F1, while the Color+Texture baseline reaches 34.44% mAP and 14.07% Macro-F1. This confirms that global color and texture descriptors are the dominant source of discriminative information for the 81-label task.

Second, Focal Loss improves the ranking quality of the best fusion model. Early Fusion reaches 36.67% mAP, which is higher than the BCE Early Fusion result of 35.16%. This suggests that Focal Loss helps the model produce better ranked probabilities for label retrieval or mAP-based evaluation.

Third, despite the higher mAP, Focal Loss produces much lower Micro-F1 and Macro-F1 than BCE under the fixed threshold of 0.5. For example, Early Fusion with Focal Loss obtains 49.07% Micro-F1 and 15.78% Macro-F1, while the BCE version obtains 60.72% Micro-F1 and 24.28% Macro-F1. This gap suggests that Focal Loss changes the probability calibration of the model. The default threshold of 0.5 may no longer be suitable, so threshold tuning on the validation set is likely necessary.

Fourth, **InfoNCE is again harmful**. Gated Fusion + InfoNCE performs worse than Gated Fusion without InfoNCE across all three metrics:

| Comparison | mAP (%) | Micro-F1 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Gated Fusion + InfoNCE | 31.72 | 38.98 | 7.35 |
| Gated Fusion (No InfoNCE) | 35.66 | 46.23 | 16.53 |

The performance drop is especially large in Macro-F1. This supports the same interpretation as in the BCE experiment: BoW and Color+Texture are heterogeneous feature types, and forcing them to align through InfoNCE may reduce useful modality-specific information.

Fifth, the validation curves show signs of overfitting or late-stage degradation. For Early Fusion, validation mAP peaks around epoch 20 and then slightly decreases by epoch 35. Gated Fusion without InfoNCE also peaks around the middle of training and then declines in validation mAP. Early stopping based on validation mAP would likely be beneficial.

### Comparison with BCE

Compared with BCE, Focal Loss gives a better best mAP:

| Loss | Best Model by mAP | Best mAP (%) |
|---|---|---:|
| BCE | Early Fusion (Concat) | 35.16 |
| Focal | Early Fusion (Concat) | **36.67** |

However, BCE is much stronger in F1-based metrics:

| Loss | Best Micro-F1 (%) | Best Macro-F1 (%) |
|---|---:|---:|
| BCE | **60.72** | **24.28** |
| Focal | 49.07 | 16.53 |

Therefore, Focal Loss appears to improve probability ranking but hurts fixed-threshold classification performance. If mAP is the primary metric, Focal Loss is promising. If Micro-F1 or Macro-F1 is the primary metric, BCE is currently better unless the prediction threshold is tuned.

### Conclusion

The best Focal Loss result is achieved by **Early Fusion (Concat)** with 36.67% mAP. This is higher than the best BCE mAP, indicating that Focal Loss can improve ranking performance for this multi-label task. However, the large drop in Micro-F1 and Macro-F1 suggests that Focal Loss is less well calibrated under a fixed 0.5 threshold. The next step should be validation-based threshold tuning, either using a global threshold or per-class thresholds. InfoNCE should not be prioritized in its current form because it consistently reduces performance under both BCE and Focal Loss.
