# ASL
```
==================================================
Training: Baseline (Local: BoW)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0153 | Val mAP: 0.1228 | Mi-F1: 0.4554 | Ma-F1: 0.1129
Epoch [02/35] Loss: 0.0142 | Val mAP: 0.1395 | Mi-F1: 0.4620 | Ma-F1: 0.1326
Epoch [03/35] Loss: 0.0139 | Val mAP: 0.1451 | Mi-F1: 0.4632 | Ma-F1: 0.1426
Epoch [04/35] Loss: 0.0137 | Val mAP: 0.1501 | Mi-F1: 0.4659 | Ma-F1: 0.1500
Epoch [05/35] Loss: 0.0134 | Val mAP: 0.1534 | Mi-F1: 0.4644 | Ma-F1: 0.1550
Epoch [06/35] Loss: 0.0132 | Val mAP: 0.1561 | Mi-F1: 0.4712 | Ma-F1: 0.1625
Epoch [07/35] Loss: 0.0130 | Val mAP: 0.1557 | Mi-F1: 0.4686 | Ma-F1: 0.1668
Epoch [08/35] Loss: 0.0128 | Val mAP: 0.1553 | Mi-F1: 0.4649 | Ma-F1: 0.1654
Epoch [09/35] Loss: 0.0126 | Val mAP: 0.1562 | Mi-F1: 0.4591 | Ma-F1: 0.1664
Epoch [10/35] Loss: 0.0124 | Val mAP: 0.1560 | Mi-F1: 0.4641 | Ma-F1: 0.1695
Epoch [11/35] Loss: 0.0122 | Val mAP: 0.1554 | Mi-F1: 0.4662 | Ma-F1: 0.1739
Epoch [12/35] Loss: 0.0120 | Val mAP: 0.1546 | Mi-F1: 0.4652 | Ma-F1: 0.1731
Epoch [13/35] Loss: 0.0119 | Val mAP: 0.1535 | Mi-F1: 0.4623 | Ma-F1: 0.1738
Epoch [14/35] Loss: 0.0117 | Val mAP: 0.1527 | Mi-F1: 0.4657 | Ma-F1: 0.1749
Epoch [15/35] Loss: 0.0115 | Val mAP: 0.1519 | Mi-F1: 0.4606 | Ma-F1: 0.1727
Epoch [16/35] Loss: 0.0114 | Val mAP: 0.1509 | Mi-F1: 0.4593 | Ma-F1: 0.1739
Epoch [17/35] Loss: 0.0113 | Val mAP: 0.1498 | Mi-F1: 0.4578 | Ma-F1: 0.1751
Epoch [18/35] Loss: 0.0112 | Val mAP: 0.1485 | Mi-F1: 0.4559 | Ma-F1: 0.1731
Epoch [19/35] Loss: 0.0110 | Val mAP: 0.1472 | Mi-F1: 0.4595 | Ma-F1: 0.1715
Epoch [20/35] Loss: 0.0110 | Val mAP: 0.1471 | Mi-F1: 0.4546 | Ma-F1: 0.1747
Epoch [21/35] Loss: 0.0108 | Val mAP: 0.1445 | Mi-F1: 0.4553 | Ma-F1: 0.1725
Epoch [22/35] Loss: 0.0108 | Val mAP: 0.1456 | Mi-F1: 0.4529 | Ma-F1: 0.1750
Epoch [23/35] Loss: 0.0107 | Val mAP: 0.1448 | Mi-F1: 0.4538 | Ma-F1: 0.1750
Epoch [24/35] Loss: 0.0106 | Val mAP: 0.1441 | Mi-F1: 0.4514 | Ma-F1: 0.1708
Epoch [25/35] Loss: 0.0105 | Val mAP: 0.1436 | Mi-F1: 0.4540 | Ma-F1: 0.1747
Epoch [26/35] Loss: 0.0104 | Val mAP: 0.1427 | Mi-F1: 0.4502 | Ma-F1: 0.1738
Epoch [27/35] Loss: 0.0104 | Val mAP: 0.1422 | Mi-F1: 0.4470 | Ma-F1: 0.1717
Epoch [28/35] Loss: 0.0103 | Val mAP: 0.1420 | Mi-F1: 0.4498 | Ma-F1: 0.1731
Epoch [29/35] Loss: 0.0102 | Val mAP: 0.1423 | Mi-F1: 0.4485 | Ma-F1: 0.1728
Epoch [30/35] Loss: 0.0102 | Val mAP: 0.1407 | Mi-F1: 0.4481 | Ma-F1: 0.1731
Epoch [31/35] Loss: 0.0101 | Val mAP: 0.1407 | Mi-F1: 0.4482 | Ma-F1: 0.1705
Epoch [32/35] Loss: 0.0101 | Val mAP: 0.1394 | Mi-F1: 0.4462 | Ma-F1: 0.1720
Epoch [33/35] Loss: 0.0100 | Val mAP: 0.1399 | Mi-F1: 0.4468 | Ma-F1: 0.1729
Epoch [34/35] Loss: 0.0100 | Val mAP: 0.1391 | Mi-F1: 0.4480 | Ma-F1: 0.1701
Epoch [35/35] Loss: 0.0099 | Val mAP: 0.1383 | Mi-F1: 0.4458 | Ma-F1: 0.1708
[TEST SET] mAP: 0.1666 | Micro-F1: 0.4788 | Macro-F1: 0.1578

==================================================
Training: Baseline (Global: Color+Texture)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0136 | Val mAP: 0.2100 | Mi-F1: 0.5329 | Ma-F1: 0.1869
Epoch [02/35] Loss: 0.0122 | Val mAP: 0.2375 | Mi-F1: 0.5490 | Ma-F1: 0.2252
Epoch [03/35] Loss: 0.0118 | Val mAP: 0.2552 | Mi-F1: 0.5559 | Ma-F1: 0.2464
Epoch [04/35] Loss: 0.0116 | Val mAP: 0.2686 | Mi-F1: 0.5587 | Ma-F1: 0.2632
Epoch [05/35] Loss: 0.0115 | Val mAP: 0.2767 | Mi-F1: 0.5600 | Ma-F1: 0.2701
Epoch [06/35] Loss: 0.0113 | Val mAP: 0.2836 | Mi-F1: 0.5659 | Ma-F1: 0.2797
Epoch [07/35] Loss: 0.0112 | Val mAP: 0.2907 | Mi-F1: 0.5715 | Ma-F1: 0.2885
Epoch [08/35] Loss: 0.0111 | Val mAP: 0.2952 | Mi-F1: 0.5710 | Ma-F1: 0.2936
Epoch [09/35] Loss: 0.0110 | Val mAP: 0.2994 | Mi-F1: 0.5735 | Ma-F1: 0.3026
Epoch [10/35] Loss: 0.0109 | Val mAP: 0.3019 | Mi-F1: 0.5736 | Ma-F1: 0.3031
Epoch [11/35] Loss: 0.0109 | Val mAP: 0.3052 | Mi-F1: 0.5767 | Ma-F1: 0.3039
Epoch [12/35] Loss: 0.0108 | Val mAP: 0.3087 | Mi-F1: 0.5764 | Ma-F1: 0.3122
Epoch [13/35] Loss: 0.0107 | Val mAP: 0.3104 | Mi-F1: 0.5731 | Ma-F1: 0.3094
Epoch [14/35] Loss: 0.0107 | Val mAP: 0.3120 | Mi-F1: 0.5797 | Ma-F1: 0.3130
Epoch [15/35] Loss: 0.0106 | Val mAP: 0.3148 | Mi-F1: 0.5812 | Ma-F1: 0.3191
Epoch [16/35] Loss: 0.0105 | Val mAP: 0.3182 | Mi-F1: 0.5775 | Ma-F1: 0.3206
Epoch [17/35] Loss: 0.0105 | Val mAP: 0.3183 | Mi-F1: 0.5794 | Ma-F1: 0.3191
Epoch [18/35] Loss: 0.0104 | Val mAP: 0.3185 | Mi-F1: 0.5805 | Ma-F1: 0.3226
Epoch [19/35] Loss: 0.0104 | Val mAP: 0.3200 | Mi-F1: 0.5814 | Ma-F1: 0.3208
Epoch [20/35] Loss: 0.0103 | Val mAP: 0.3215 | Mi-F1: 0.5832 | Ma-F1: 0.3256
Epoch [21/35] Loss: 0.0103 | Val mAP: 0.3213 | Mi-F1: 0.5844 | Ma-F1: 0.3247
Epoch [22/35] Loss: 0.0102 | Val mAP: 0.3228 | Mi-F1: 0.5836 | Ma-F1: 0.3248
Epoch [23/35] Loss: 0.0102 | Val mAP: 0.3248 | Mi-F1: 0.5826 | Ma-F1: 0.3246
Epoch [24/35] Loss: 0.0101 | Val mAP: 0.3246 | Mi-F1: 0.5812 | Ma-F1: 0.3277
Epoch [25/35] Loss: 0.0101 | Val mAP: 0.3267 | Mi-F1: 0.5818 | Ma-F1: 0.3282
Epoch [26/35] Loss: 0.0100 | Val mAP: 0.3261 | Mi-F1: 0.5834 | Ma-F1: 0.3292
Epoch [27/35] Loss: 0.0100 | Val mAP: 0.3276 | Mi-F1: 0.5862 | Ma-F1: 0.3314
Epoch [28/35] Loss: 0.0100 | Val mAP: 0.3287 | Mi-F1: 0.5836 | Ma-F1: 0.3309
Epoch [29/35] Loss: 0.0099 | Val mAP: 0.3279 | Mi-F1: 0.5851 | Ma-F1: 0.3342
Epoch [30/35] Loss: 0.0099 | Val mAP: 0.3286 | Mi-F1: 0.5850 | Ma-F1: 0.3319
Epoch [31/35] Loss: 0.0098 | Val mAP: 0.3288 | Mi-F1: 0.5837 | Ma-F1: 0.3317
Epoch [32/35] Loss: 0.0098 | Val mAP: 0.3285 | Mi-F1: 0.5855 | Ma-F1: 0.3333
Epoch [33/35] Loss: 0.0098 | Val mAP: 0.3294 | Mi-F1: 0.5829 | Ma-F1: 0.3336
Epoch [34/35] Loss: 0.0097 | Val mAP: 0.3296 | Mi-F1: 0.5841 | Ma-F1: 0.3339
Epoch [35/35] Loss: 0.0097 | Val mAP: 0.3297 | Mi-F1: 0.5851 | Ma-F1: 0.3362
[TEST SET] mAP: 0.3502 | Micro-F1: 0.6347 | Macro-F1: 0.3275

==================================================
Training: Early Fusion (Concat)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0135 | Val mAP: 0.2028 | Mi-F1: 0.5386 | Ma-F1: 0.1829
Epoch [02/35] Loss: 0.0120 | Val mAP: 0.2371 | Mi-F1: 0.5588 | Ma-F1: 0.2302
Epoch [03/35] Loss: 0.0115 | Val mAP: 0.2543 | Mi-F1: 0.5625 | Ma-F1: 0.2423
Epoch [04/35] Loss: 0.0112 | Val mAP: 0.2662 | Mi-F1: 0.5726 | Ma-F1: 0.2579
Epoch [05/35] Loss: 0.0110 | Val mAP: 0.2767 | Mi-F1: 0.5733 | Ma-F1: 0.2726
Epoch [06/35] Loss: 0.0108 | Val mAP: 0.2821 | Mi-F1: 0.5767 | Ma-F1: 0.2829
Epoch [07/35] Loss: 0.0107 | Val mAP: 0.2889 | Mi-F1: 0.5807 | Ma-F1: 0.2896
Epoch [08/35] Loss: 0.0105 | Val mAP: 0.2920 | Mi-F1: 0.5808 | Ma-F1: 0.2938
Epoch [09/35] Loss: 0.0104 | Val mAP: 0.2960 | Mi-F1: 0.5789 | Ma-F1: 0.3012
Epoch [10/35] Loss: 0.0103 | Val mAP: 0.2991 | Mi-F1: 0.5805 | Ma-F1: 0.3045
Epoch [11/35] Loss: 0.0101 | Val mAP: 0.3013 | Mi-F1: 0.5753 | Ma-F1: 0.3079
Epoch [12/35] Loss: 0.0100 | Val mAP: 0.3025 | Mi-F1: 0.5850 | Ma-F1: 0.3066
Epoch [13/35] Loss: 0.0099 | Val mAP: 0.3038 | Mi-F1: 0.5822 | Ma-F1: 0.3126
Epoch [14/35] Loss: 0.0098 | Val mAP: 0.3048 | Mi-F1: 0.5826 | Ma-F1: 0.3142
Epoch [15/35] Loss: 0.0097 | Val mAP: 0.3073 | Mi-F1: 0.5824 | Ma-F1: 0.3162
Epoch [16/35] Loss: 0.0096 | Val mAP: 0.3070 | Mi-F1: 0.5795 | Ma-F1: 0.3178
Epoch [17/35] Loss: 0.0095 | Val mAP: 0.3062 | Mi-F1: 0.5809 | Ma-F1: 0.3166
Epoch [18/35] Loss: 0.0094 | Val mAP: 0.3084 | Mi-F1: 0.5823 | Ma-F1: 0.3195
Epoch [19/35] Loss: 0.0093 | Val mAP: 0.3080 | Mi-F1: 0.5768 | Ma-F1: 0.3203
Epoch [20/35] Loss: 0.0092 | Val mAP: 0.3097 | Mi-F1: 0.5803 | Ma-F1: 0.3201
Epoch [21/35] Loss: 0.0092 | Val mAP: 0.3104 | Mi-F1: 0.5770 | Ma-F1: 0.3213
Epoch [22/35] Loss: 0.0091 | Val mAP: 0.3091 | Mi-F1: 0.5806 | Ma-F1: 0.3219
Epoch [23/35] Loss: 0.0090 | Val mAP: 0.3086 | Mi-F1: 0.5768 | Ma-F1: 0.3199
Epoch [24/35] Loss: 0.0089 | Val mAP: 0.3067 | Mi-F1: 0.5800 | Ma-F1: 0.3224
Epoch [25/35] Loss: 0.0089 | Val mAP: 0.3095 | Mi-F1: 0.5819 | Ma-F1: 0.3258
Epoch [26/35] Loss: 0.0088 | Val mAP: 0.3099 | Mi-F1: 0.5787 | Ma-F1: 0.3233
Epoch [27/35] Loss: 0.0087 | Val mAP: 0.3080 | Mi-F1: 0.5787 | Ma-F1: 0.3275
Epoch [28/35] Loss: 0.0087 | Val mAP: 0.3086 | Mi-F1: 0.5781 | Ma-F1: 0.3242
Epoch [29/35] Loss: 0.0086 | Val mAP: 0.3062 | Mi-F1: 0.5776 | Ma-F1: 0.3198
Epoch [30/35] Loss: 0.0085 | Val mAP: 0.3068 | Mi-F1: 0.5745 | Ma-F1: 0.3223
Epoch [31/35] Loss: 0.0085 | Val mAP: 0.3074 | Mi-F1: 0.5766 | Ma-F1: 0.3225
Epoch [32/35] Loss: 0.0084 | Val mAP: 0.3056 | Mi-F1: 0.5741 | Ma-F1: 0.3206
Epoch [33/35] Loss: 0.0084 | Val mAP: 0.3059 | Mi-F1: 0.5750 | Ma-F1: 0.3222
Epoch [34/35] Loss: 0.0083 | Val mAP: 0.3064 | Mi-F1: 0.5757 | Ma-F1: 0.3205
Epoch [35/35] Loss: 0.0083 | Val mAP: 0.3035 | Mi-F1: 0.5766 | Ma-F1: 0.3220
[TEST SET] mAP: 0.3577 | Micro-F1: 0.6244 | Macro-F1: 0.3322

==================================================
Training: Gated Fusion + InfoNCE
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.1776 | Val mAP: 0.1431 | Mi-F1: 0.4880 | Ma-F1: 0.1194
Epoch [02/35] Loss: 0.1275 | Val mAP: 0.1736 | Mi-F1: 0.5139 | Ma-F1: 0.1623
Epoch [03/35] Loss: 0.1116 | Val mAP: 0.1931 | Mi-F1: 0.5254 | Ma-F1: 0.1856
Epoch [04/35] Loss: 0.1022 | Val mAP: 0.2087 | Mi-F1: 0.5373 | Ma-F1: 0.1979
Epoch [05/35] Loss: 0.0962 | Val mAP: 0.2193 | Mi-F1: 0.5380 | Ma-F1: 0.2142
Epoch [06/35] Loss: 0.0923 | Val mAP: 0.2260 | Mi-F1: 0.5488 | Ma-F1: 0.2196
Epoch [07/35] Loss: 0.0893 | Val mAP: 0.2323 | Mi-F1: 0.5479 | Ma-F1: 0.2298
Epoch [08/35] Loss: 0.0872 | Val mAP: 0.2385 | Mi-F1: 0.5529 | Ma-F1: 0.2306
Epoch [09/35] Loss: 0.0856 | Val mAP: 0.2424 | Mi-F1: 0.5517 | Ma-F1: 0.2416
Epoch [10/35] Loss: 0.0843 | Val mAP: 0.2475 | Mi-F1: 0.5576 | Ma-F1: 0.2467
Epoch [11/35] Loss: 0.0830 | Val mAP: 0.2516 | Mi-F1: 0.5577 | Ma-F1: 0.2458
Epoch [12/35] Loss: 0.0819 | Val mAP: 0.2534 | Mi-F1: 0.5542 | Ma-F1: 0.2511
Epoch [13/35] Loss: 0.0812 | Val mAP: 0.2558 | Mi-F1: 0.5593 | Ma-F1: 0.2532
Epoch [14/35] Loss: 0.0804 | Val mAP: 0.2586 | Mi-F1: 0.5615 | Ma-F1: 0.2587
Epoch [15/35] Loss: 0.0795 | Val mAP: 0.2592 | Mi-F1: 0.5627 | Ma-F1: 0.2600
Epoch [16/35] Loss: 0.0790 | Val mAP: 0.2627 | Mi-F1: 0.5634 | Ma-F1: 0.2668
Epoch [17/35] Loss: 0.0782 | Val mAP: 0.2655 | Mi-F1: 0.5663 | Ma-F1: 0.2685
Epoch [18/35] Loss: 0.0776 | Val mAP: 0.2670 | Mi-F1: 0.5631 | Ma-F1: 0.2711
Epoch [19/35] Loss: 0.0773 | Val mAP: 0.2678 | Mi-F1: 0.5646 | Ma-F1: 0.2685
Epoch [20/35] Loss: 0.0767 | Val mAP: 0.2688 | Mi-F1: 0.5641 | Ma-F1: 0.2720
Epoch [21/35] Loss: 0.0763 | Val mAP: 0.2712 | Mi-F1: 0.5669 | Ma-F1: 0.2765
Epoch [22/35] Loss: 0.0760 | Val mAP: 0.2724 | Mi-F1: 0.5657 | Ma-F1: 0.2746
Epoch [23/35] Loss: 0.0755 | Val mAP: 0.2722 | Mi-F1: 0.5666 | Ma-F1: 0.2764
Epoch [24/35] Loss: 0.0753 | Val mAP: 0.2736 | Mi-F1: 0.5665 | Ma-F1: 0.2792
Epoch [25/35] Loss: 0.0748 | Val mAP: 0.2747 | Mi-F1: 0.5664 | Ma-F1: 0.2771
Epoch [26/35] Loss: 0.0745 | Val mAP: 0.2758 | Mi-F1: 0.5649 | Ma-F1: 0.2793
Epoch [27/35] Loss: 0.0742 | Val mAP: 0.2779 | Mi-F1: 0.5678 | Ma-F1: 0.2812
Epoch [28/35] Loss: 0.0739 | Val mAP: 0.2793 | Mi-F1: 0.5697 | Ma-F1: 0.2838
Epoch [29/35] Loss: 0.0737 | Val mAP: 0.2781 | Mi-F1: 0.5696 | Ma-F1: 0.2811
Epoch [30/35] Loss: 0.0735 | Val mAP: 0.2788 | Mi-F1: 0.5685 | Ma-F1: 0.2804
Epoch [31/35] Loss: 0.0733 | Val mAP: 0.2817 | Mi-F1: 0.5672 | Ma-F1: 0.2857
Epoch [32/35] Loss: 0.0730 | Val mAP: 0.2823 | Mi-F1: 0.5681 | Ma-F1: 0.2869
Epoch [33/35] Loss: 0.0728 | Val mAP: 0.2818 | Mi-F1: 0.5685 | Ma-F1: 0.2868
Epoch [34/35] Loss: 0.0726 | Val mAP: 0.2811 | Mi-F1: 0.5688 | Ma-F1: 0.2881
Epoch [35/35] Loss: 0.0724 | Val mAP: 0.2845 | Mi-F1: 0.5704 | Ma-F1: 0.2900
[TEST SET] mAP: 0.3297 | Micro-F1: 0.6179 | Macro-F1: 0.2952

==================================================
Training: Gated Fusion (No InfoNCE)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0136 | Val mAP: 0.1996 | Mi-F1: 0.5339 | Ma-F1: 0.1714
Epoch [02/35] Loss: 0.0120 | Val mAP: 0.2336 | Mi-F1: 0.5600 | Ma-F1: 0.2187
Epoch [03/35] Loss: 0.0116 | Val mAP: 0.2524 | Mi-F1: 0.5688 | Ma-F1: 0.2458
Epoch [04/35] Loss: 0.0113 | Val mAP: 0.2649 | Mi-F1: 0.5705 | Ma-F1: 0.2560
Epoch [05/35] Loss: 0.0111 | Val mAP: 0.2744 | Mi-F1: 0.5690 | Ma-F1: 0.2700
Epoch [06/35] Loss: 0.0109 | Val mAP: 0.2809 | Mi-F1: 0.5694 | Ma-F1: 0.2759
Epoch [07/35] Loss: 0.0108 | Val mAP: 0.2869 | Mi-F1: 0.5745 | Ma-F1: 0.2860
Epoch [08/35] Loss: 0.0106 | Val mAP: 0.2902 | Mi-F1: 0.5754 | Ma-F1: 0.2905
Epoch [09/35] Loss: 0.0105 | Val mAP: 0.2933 | Mi-F1: 0.5780 | Ma-F1: 0.2931
Epoch [10/35] Loss: 0.0104 | Val mAP: 0.2962 | Mi-F1: 0.5819 | Ma-F1: 0.2973
Epoch [11/35] Loss: 0.0103 | Val mAP: 0.2986 | Mi-F1: 0.5811 | Ma-F1: 0.3040
Epoch [12/35] Loss: 0.0102 | Val mAP: 0.3003 | Mi-F1: 0.5824 | Ma-F1: 0.3050
Epoch [13/35] Loss: 0.0101 | Val mAP: 0.3017 | Mi-F1: 0.5808 | Ma-F1: 0.3064
Epoch [14/35] Loss: 0.0100 | Val mAP: 0.3024 | Mi-F1: 0.5817 | Ma-F1: 0.3073
Epoch [15/35] Loss: 0.0099 | Val mAP: 0.3019 | Mi-F1: 0.5798 | Ma-F1: 0.3087
Epoch [16/35] Loss: 0.0098 | Val mAP: 0.3041 | Mi-F1: 0.5761 | Ma-F1: 0.3115
Epoch [17/35] Loss: 0.0098 | Val mAP: 0.3040 | Mi-F1: 0.5759 | Ma-F1: 0.3136
Epoch [18/35] Loss: 0.0097 | Val mAP: 0.3053 | Mi-F1: 0.5820 | Ma-F1: 0.3158
Epoch [19/35] Loss: 0.0096 | Val mAP: 0.3055 | Mi-F1: 0.5838 | Ma-F1: 0.3151
Epoch [20/35] Loss: 0.0095 | Val mAP: 0.3049 | Mi-F1: 0.5781 | Ma-F1: 0.3163
Epoch [21/35] Loss: 0.0095 | Val mAP: 0.3044 | Mi-F1: 0.5739 | Ma-F1: 0.3148
Epoch [22/35] Loss: 0.0094 | Val mAP: 0.3062 | Mi-F1: 0.5759 | Ma-F1: 0.3185
Epoch [23/35] Loss: 0.0093 | Val mAP: 0.3048 | Mi-F1: 0.5796 | Ma-F1: 0.3197
Epoch [24/35] Loss: 0.0093 | Val mAP: 0.3035 | Mi-F1: 0.5738 | Ma-F1: 0.3147
Epoch [25/35] Loss: 0.0092 | Val mAP: 0.3043 | Mi-F1: 0.5722 | Ma-F1: 0.3177
Epoch [26/35] Loss: 0.0092 | Val mAP: 0.3048 | Mi-F1: 0.5718 | Ma-F1: 0.3182
Epoch [27/35] Loss: 0.0091 | Val mAP: 0.3039 | Mi-F1: 0.5724 | Ma-F1: 0.3149
Epoch [28/35] Loss: 0.0091 | Val mAP: 0.3034 | Mi-F1: 0.5724 | Ma-F1: 0.3174
Epoch [29/35] Loss: 0.0090 | Val mAP: 0.3021 | Mi-F1: 0.5757 | Ma-F1: 0.3161
Epoch [30/35] Loss: 0.0089 | Val mAP: 0.3019 | Mi-F1: 0.5763 | Ma-F1: 0.3194
Epoch [31/35] Loss: 0.0089 | Val mAP: 0.3005 | Mi-F1: 0.5701 | Ma-F1: 0.3145
Epoch [32/35] Loss: 0.0088 | Val mAP: 0.3004 | Mi-F1: 0.5747 | Ma-F1: 0.3148
Epoch [33/35] Loss: 0.0088 | Val mAP: 0.2992 | Mi-F1: 0.5720 | Ma-F1: 0.3164
Epoch [34/35] Loss: 0.0087 | Val mAP: 0.3012 | Mi-F1: 0.5714 | Ma-F1: 0.3172
Epoch [35/35] Loss: 0.0087 | Val mAP: 0.2994 | Mi-F1: 0.5713 | Ma-F1: 0.3155
[TEST SET] mAP: 0.3486 | Micro-F1: 0.6332 | Macro-F1: 0.3398

====================================================================
Experiment Model               | mAP (%)  | Micro-F1 | Macro-F1
--------------------------------------------------------------------
Baseline_A (BoW)               | 16.66     | 47.88     | 15.78
Baseline_B (Color+Tex)         | 35.02     | 63.47     | 32.75
Early_Fusion (Concat)          | 35.77     | 62.44     | 33.22
Gated_Fusion + InfoNCE         | 32.97     | 61.79     | 29.52
Gated_Fusion (No NCE)          | 34.86     | 63.32     | 33.98
====================================================================
```

## Analysis

The ASL experiments show a different behavior from BCE and Focal Loss. **Early Fusion (Concat)** achieves the best mAP, while **Gated Fusion (No InfoNCE)** achieves the best Macro-F1:

| Model | mAP (%) | Micro-F1 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Baseline_A (BoW) | 16.66 | 47.88 | 15.78 |
| Baseline_B (Color+Texture) | 35.02 | **63.47** | 32.75 |
| Early_Fusion (Concat) | **35.77** | 62.44 | 33.22 |
| Gated_Fusion + InfoNCE | 32.97 | 61.79 | 29.52 |
| Gated_Fusion (No InfoNCE) | 34.86 | 63.32 | **33.98** |

### Main Findings

First, the **Color+Texture** features remain much stronger than the **BoW** features. The BoW-only model obtains only 16.66% mAP, while the Color+Texture baseline reaches 35.02% mAP. This is consistent with the BCE and Focal experiments: global visual descriptors are the main source of predictive power for this 81-label task.

Second, ASL gives a very large improvement in Macro-F1 compared with both BCE and Focal Loss. The best Macro-F1 under ASL is 33.98%, achieved by Gated Fusion without InfoNCE. This is much higher than the best BCE Macro-F1 of 24.28% and the best Focal Macro-F1 of 16.53%. This suggests that ASL is much better at handling the class imbalance and the large number of negative labels in this multi-label setting.

Third, the best mAP is achieved by Early Fusion with 35.77%. This is slightly higher than the best BCE mAP of 35.16%, but lower than the best Focal mAP of 36.67%. Therefore, ASL is not the strongest loss for mAP in these experiments, but it gives the best balance for F1-based classification metrics, especially Macro-F1.

Fourth, **InfoNCE is again harmful**. Gated Fusion + InfoNCE is worse than Gated Fusion without InfoNCE across all metrics:

| Comparison | mAP (%) | Micro-F1 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Gated Fusion + InfoNCE | 32.97 | 61.79 | 29.52 |
| Gated Fusion (No InfoNCE) | 34.86 | 63.32 | 33.98 |

This confirms the pattern observed in both BCE and Focal Loss. The alignment objective does not seem suitable for these heterogeneous features. BoW and Color+Texture likely encode different kinds of information, and forcing their latent representations to match may reduce useful modality-specific signals.

Fifth, the fusion results show that simple fusion is more useful for mAP, while gated fusion is more useful for Macro-F1. Early Fusion has the highest mAP, but Gated Fusion without InfoNCE has the highest Macro-F1. This suggests that the gate may help balance predictions across labels, including less frequent classes, even if it does not produce the best ranking score.

### Comparison with BCE and Focal Loss

Across the three losses, the best mAP is obtained by Focal Loss, while the best Micro-F1 and Macro-F1 are obtained by ASL:

| Loss | Best Model by mAP | Best mAP (%) |
|---|---|---:|
| BCE | Early Fusion (Concat) | 35.16 |
| Focal | Early Fusion (Concat) | **36.67** |
| ASL | Early Fusion (Concat) | 35.77 |

| Loss | Best Micro-F1 (%) | Best Macro-F1 (%) |
|---|---:|---:|
| BCE | 60.72 | 24.28 |
| Focal | 49.07 | 16.53 |
| ASL | **63.47** | **33.98** |

These results indicate that Focal Loss improves probability ranking, but ASL is much better for threshold-based multi-label classification. Since Macro-F1 is especially sensitive to minority classes, the strong ASL Macro-F1 suggests that it is the most effective loss for handling label imbalance in this experiment.

### Conclusion

ASL gives the strongest overall classification performance among the three loss functions tested so far. Although Focal Loss achieves the highest mAP, ASL clearly outperforms both BCE and Focal Loss in Micro-F1 and Macro-F1. The best ASL configuration is **Gated Fusion (No InfoNCE)** for Macro-F1 and **Early Fusion (Concat)** for mAP. InfoNCE should not be prioritized in its current form, because it consistently reduces performance under BCE, Focal Loss, and ASL. For the final model choice, ASL with either Early Fusion or Gated Fusion without InfoNCE is the most promising direction, depending on whether the target metric is mAP or Macro-F1.
