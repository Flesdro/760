# BCE
```
==================================================
Training: Baseline (Local: BoW)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0947 | Val mAP: 0.1133 | Mi-F1: 0.3562 | Ma-F1: 0.0329
Epoch [02/35] Loss: 0.0830 | Val mAP: 0.1284 | Mi-F1: 0.3778 | Ma-F1: 0.0451
Epoch [03/35] Loss: 0.0807 | Val mAP: 0.1361 | Mi-F1: 0.3812 | Ma-F1: 0.0508
Epoch [04/35] Loss: 0.0791 | Val mAP: 0.1419 | Mi-F1: 0.4010 | Ma-F1: 0.0587
Epoch [05/35] Loss: 0.0779 | Val mAP: 0.1458 | Mi-F1: 0.4107 | Ma-F1: 0.0662
Epoch [06/35] Loss: 0.0766 | Val mAP: 0.1478 | Mi-F1: 0.4111 | Ma-F1: 0.0696
Epoch [07/35] Loss: 0.0753 | Val mAP: 0.1469 | Mi-F1: 0.4101 | Ma-F1: 0.0688
Epoch [08/35] Loss: 0.0743 | Val mAP: 0.1498 | Mi-F1: 0.4162 | Ma-F1: 0.0764
Epoch [09/35] Loss: 0.0730 | Val mAP: 0.1503 | Mi-F1: 0.4216 | Ma-F1: 0.0791
Epoch [10/35] Loss: 0.0718 | Val mAP: 0.1506 | Mi-F1: 0.4243 | Ma-F1: 0.0841
Epoch [11/35] Loss: 0.0706 | Val mAP: 0.1513 | Mi-F1: 0.4289 | Ma-F1: 0.0853
Epoch [12/35] Loss: 0.0696 | Val mAP: 0.1502 | Mi-F1: 0.4162 | Ma-F1: 0.0844
Epoch [13/35] Loss: 0.0686 | Val mAP: 0.1501 | Mi-F1: 0.4219 | Ma-F1: 0.0887
Epoch [14/35] Loss: 0.0676 | Val mAP: 0.1488 | Mi-F1: 0.4238 | Ma-F1: 0.0883
Epoch [15/35] Loss: 0.0667 | Val mAP: 0.1478 | Mi-F1: 0.4236 | Ma-F1: 0.0891
Epoch [16/35] Loss: 0.0658 | Val mAP: 0.1472 | Mi-F1: 0.4258 | Ma-F1: 0.0898
Epoch [17/35] Loss: 0.0650 | Val mAP: 0.1473 | Mi-F1: 0.4261 | Ma-F1: 0.0924
Epoch [18/35] Loss: 0.0642 | Val mAP: 0.1456 | Mi-F1: 0.4300 | Ma-F1: 0.0955
Epoch [19/35] Loss: 0.0635 | Val mAP: 0.1433 | Mi-F1: 0.4247 | Ma-F1: 0.0913
Epoch [20/35] Loss: 0.0629 | Val mAP: 0.1429 | Mi-F1: 0.4195 | Ma-F1: 0.0912
Epoch [21/35] Loss: 0.0623 | Val mAP: 0.1420 | Mi-F1: 0.4255 | Ma-F1: 0.0972
Epoch [22/35] Loss: 0.0615 | Val mAP: 0.1404 | Mi-F1: 0.4215 | Ma-F1: 0.0920
Epoch [23/35] Loss: 0.0610 | Val mAP: 0.1401 | Mi-F1: 0.4280 | Ma-F1: 0.0940
Epoch [24/35] Loss: 0.0605 | Val mAP: 0.1410 | Mi-F1: 0.4282 | Ma-F1: 0.0973
Epoch [25/35] Loss: 0.0599 | Val mAP: 0.1394 | Mi-F1: 0.4291 | Ma-F1: 0.0973
Epoch [26/35] Loss: 0.0595 | Val mAP: 0.1378 | Mi-F1: 0.4237 | Ma-F1: 0.0985
Epoch [27/35] Loss: 0.0595 | Val mAP: 0.1383 | Mi-F1: 0.4232 | Ma-F1: 0.1001
Epoch [28/35] Loss: 0.0586 | Val mAP: 0.1370 | Mi-F1: 0.4220 | Ma-F1: 0.0971
Epoch [29/35] Loss: 0.0581 | Val mAP: 0.1369 | Mi-F1: 0.4294 | Ma-F1: 0.1009
Epoch [30/35] Loss: 0.0578 | Val mAP: 0.1351 | Mi-F1: 0.4229 | Ma-F1: 0.0965
Epoch [31/35] Loss: 0.0575 | Val mAP: 0.1370 | Mi-F1: 0.4270 | Ma-F1: 0.1009
Epoch [32/35] Loss: 0.0570 | Val mAP: 0.1351 | Mi-F1: 0.4267 | Ma-F1: 0.0994
Epoch [33/35] Loss: 0.0567 | Val mAP: 0.1356 | Mi-F1: 0.4250 | Ma-F1: 0.0985
Epoch [34/35] Loss: 0.0563 | Val mAP: 0.1341 | Mi-F1: 0.4254 | Ma-F1: 0.0991
Epoch [35/35] Loss: 0.0561 | Val mAP: 0.1316 | Mi-F1: 0.4223 | Ma-F1: 0.0997
[TEST SET] mAP: 0.1746 | Micro-F1: 0.4092 | Macro-F1: 0.0940

==================================================
Training: Baseline (Global: Color+Texture)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0922 | Val mAP: 0.1722 | Mi-F1: 0.4851 | Ma-F1: 0.0730
Epoch [02/35] Loss: 0.0717 | Val mAP: 0.2059 | Mi-F1: 0.5183 | Ma-F1: 0.0956
Epoch [03/35] Loss: 0.0689 | Val mAP: 0.2250 | Mi-F1: 0.5331 | Ma-F1: 0.1066
Epoch [04/35] Loss: 0.0672 | Val mAP: 0.2404 | Mi-F1: 0.5431 | Ma-F1: 0.1258
Epoch [05/35] Loss: 0.0660 | Val mAP: 0.2510 | Mi-F1: 0.5529 | Ma-F1: 0.1320
Epoch [06/35] Loss: 0.0651 | Val mAP: 0.2607 | Mi-F1: 0.5597 | Ma-F1: 0.1404
Epoch [07/35] Loss: 0.0643 | Val mAP: 0.2685 | Mi-F1: 0.5638 | Ma-F1: 0.1505
Epoch [08/35] Loss: 0.0636 | Val mAP: 0.2741 | Mi-F1: 0.5659 | Ma-F1: 0.1588
Epoch [09/35] Loss: 0.0629 | Val mAP: 0.2797 | Mi-F1: 0.5713 | Ma-F1: 0.1625
Epoch [10/35] Loss: 0.0624 | Val mAP: 0.2808 | Mi-F1: 0.5773 | Ma-F1: 0.1709
Epoch [11/35] Loss: 0.0619 | Val mAP: 0.2871 | Mi-F1: 0.5782 | Ma-F1: 0.1714
Epoch [12/35] Loss: 0.0614 | Val mAP: 0.2905 | Mi-F1: 0.5793 | Ma-F1: 0.1724
Epoch [13/35] Loss: 0.0610 | Val mAP: 0.2935 | Mi-F1: 0.5843 | Ma-F1: 0.1839
Epoch [14/35] Loss: 0.0606 | Val mAP: 0.2964 | Mi-F1: 0.5823 | Ma-F1: 0.1891
Epoch [15/35] Loss: 0.0601 | Val mAP: 0.2990 | Mi-F1: 0.5846 | Ma-F1: 0.1883
Epoch [16/35] Loss: 0.0597 | Val mAP: 0.3020 | Mi-F1: 0.5918 | Ma-F1: 0.1981
Epoch [17/35] Loss: 0.0594 | Val mAP: 0.3036 | Mi-F1: 0.5891 | Ma-F1: 0.1991
Epoch [18/35] Loss: 0.0590 | Val mAP: 0.3063 | Mi-F1: 0.5907 | Ma-F1: 0.2025
Epoch [19/35] Loss: 0.0587 | Val mAP: 0.3081 | Mi-F1: 0.5940 | Ma-F1: 0.2066
Epoch [20/35] Loss: 0.0584 | Val mAP: 0.3087 | Mi-F1: 0.5945 | Ma-F1: 0.2101
Epoch [21/35] Loss: 0.0581 | Val mAP: 0.3109 | Mi-F1: 0.5956 | Ma-F1: 0.2133
Epoch [22/35] Loss: 0.0577 | Val mAP: 0.3121 | Mi-F1: 0.5943 | Ma-F1: 0.2142
Epoch [23/35] Loss: 0.0574 | Val mAP: 0.3133 | Mi-F1: 0.5967 | Ma-F1: 0.2220
Epoch [24/35] Loss: 0.0571 | Val mAP: 0.3145 | Mi-F1: 0.5986 | Ma-F1: 0.2199
Epoch [25/35] Loss: 0.0568 | Val mAP: 0.3166 | Mi-F1: 0.5995 | Ma-F1: 0.2232
Epoch [26/35] Loss: 0.0565 | Val mAP: 0.3164 | Mi-F1: 0.5973 | Ma-F1: 0.2239
Epoch [27/35] Loss: 0.0563 | Val mAP: 0.3187 | Mi-F1: 0.5986 | Ma-F1: 0.2258
Epoch [28/35] Loss: 0.0560 | Val mAP: 0.3183 | Mi-F1: 0.6006 | Ma-F1: 0.2272
Epoch [29/35] Loss: 0.0557 | Val mAP: 0.3184 | Mi-F1: 0.5999 | Ma-F1: 0.2303
Epoch [30/35] Loss: 0.0555 | Val mAP: 0.3195 | Mi-F1: 0.6004 | Ma-F1: 0.2272
Epoch [31/35] Loss: 0.0552 | Val mAP: 0.3201 | Mi-F1: 0.6034 | Ma-F1: 0.2333
Epoch [32/35] Loss: 0.0550 | Val mAP: 0.3212 | Mi-F1: 0.6034 | Ma-F1: 0.2327
Epoch [33/35] Loss: 0.0547 | Val mAP: 0.3211 | Mi-F1: 0.6058 | Ma-F1: 0.2354
Epoch [34/35] Loss: 0.0545 | Val mAP: 0.3239 | Mi-F1: 0.6033 | Ma-F1: 0.2346
Epoch [35/35] Loss: 0.0543 | Val mAP: 0.3221 | Mi-F1: 0.6030 | Ma-F1: 0.2364
[TEST SET] mAP: 0.3490 | Micro-F1: 0.6018 | Macro-F1: 0.2288

==================================================
Training: Early Fusion (Concat)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0861 | Val mAP: 0.1780 | Mi-F1: 0.5148 | Ma-F1: 0.0842
Epoch [02/35] Loss: 0.0701 | Val mAP: 0.2143 | Mi-F1: 0.5322 | Ma-F1: 0.1053
Epoch [03/35] Loss: 0.0666 | Val mAP: 0.2359 | Mi-F1: 0.5578 | Ma-F1: 0.1247
Epoch [04/35] Loss: 0.0644 | Val mAP: 0.2505 | Mi-F1: 0.5681 | Ma-F1: 0.1416
Epoch [05/35] Loss: 0.0629 | Val mAP: 0.2626 | Mi-F1: 0.5752 | Ma-F1: 0.1600
Epoch [06/35] Loss: 0.0617 | Val mAP: 0.2706 | Mi-F1: 0.5739 | Ma-F1: 0.1656
Epoch [07/35] Loss: 0.0606 | Val mAP: 0.2761 | Mi-F1: 0.5793 | Ma-F1: 0.1694
Epoch [08/35] Loss: 0.0597 | Val mAP: 0.2826 | Mi-F1: 0.5851 | Ma-F1: 0.1800
Epoch [09/35] Loss: 0.0588 | Val mAP: 0.2877 | Mi-F1: 0.5912 | Ma-F1: 0.1888
Epoch [10/35] Loss: 0.0580 | Val mAP: 0.2900 | Mi-F1: 0.5925 | Ma-F1: 0.1949
Epoch [11/35] Loss: 0.0572 | Val mAP: 0.2945 | Mi-F1: 0.5919 | Ma-F1: 0.1984
Epoch [12/35] Loss: 0.0565 | Val mAP: 0.2977 | Mi-F1: 0.5915 | Ma-F1: 0.2040
Epoch [13/35] Loss: 0.0558 | Val mAP: 0.2986 | Mi-F1: 0.5947 | Ma-F1: 0.2058
Epoch [14/35] Loss: 0.0552 | Val mAP: 0.3019 | Mi-F1: 0.5955 | Ma-F1: 0.2136
Epoch [15/35] Loss: 0.0546 | Val mAP: 0.3031 | Mi-F1: 0.5954 | Ma-F1: 0.2096
Epoch [16/35] Loss: 0.0539 | Val mAP: 0.3036 | Mi-F1: 0.5961 | Ma-F1: 0.2192
Epoch [17/35] Loss: 0.0534 | Val mAP: 0.3039 | Mi-F1: 0.5985 | Ma-F1: 0.2229
Epoch [18/35] Loss: 0.0528 | Val mAP: 0.3048 | Mi-F1: 0.5987 | Ma-F1: 0.2235
Epoch [19/35] Loss: 0.0523 | Val mAP: 0.3030 | Mi-F1: 0.6007 | Ma-F1: 0.2307
Epoch [20/35] Loss: 0.0518 | Val mAP: 0.3041 | Mi-F1: 0.5990 | Ma-F1: 0.2293
Epoch [21/35] Loss: 0.0513 | Val mAP: 0.3047 | Mi-F1: 0.6018 | Ma-F1: 0.2395
Epoch [22/35] Loss: 0.0508 | Val mAP: 0.3053 | Mi-F1: 0.5996 | Ma-F1: 0.2399
Epoch [23/35] Loss: 0.0504 | Val mAP: 0.3049 | Mi-F1: 0.6007 | Ma-F1: 0.2458
Epoch [24/35] Loss: 0.0499 | Val mAP: 0.3050 | Mi-F1: 0.5997 | Ma-F1: 0.2404
Epoch [25/35] Loss: 0.0495 | Val mAP: 0.3034 | Mi-F1: 0.5979 | Ma-F1: 0.2447
Epoch [26/35] Loss: 0.0491 | Val mAP: 0.3018 | Mi-F1: 0.5975 | Ma-F1: 0.2502
Epoch [27/35] Loss: 0.0487 | Val mAP: 0.3045 | Mi-F1: 0.6002 | Ma-F1: 0.2454
Epoch [28/35] Loss: 0.0483 | Val mAP: 0.3050 | Mi-F1: 0.5976 | Ma-F1: 0.2482
Epoch [29/35] Loss: 0.0478 | Val mAP: 0.3032 | Mi-F1: 0.5998 | Ma-F1: 0.2500
Epoch [30/35] Loss: 0.0475 | Val mAP: 0.3030 | Mi-F1: 0.5990 | Ma-F1: 0.2548
Epoch [31/35] Loss: 0.0471 | Val mAP: 0.3012 | Mi-F1: 0.5997 | Ma-F1: 0.2556
Epoch [32/35] Loss: 0.0468 | Val mAP: 0.3030 | Mi-F1: 0.6002 | Ma-F1: 0.2557
Epoch [33/35] Loss: 0.0465 | Val mAP: 0.3023 | Mi-F1: 0.5995 | Ma-F1: 0.2552
Epoch [34/35] Loss: 0.0461 | Val mAP: 0.3010 | Mi-F1: 0.5990 | Ma-F1: 0.2547
Epoch [35/35] Loss: 0.0458 | Val mAP: 0.2953 | Mi-F1: 0.5992 | Ma-F1: 0.2538
[TEST SET] mAP: 0.3516 | Micro-F1: 0.6072 | Macro-F1: 0.2428

==================================================
Training: Gated Fusion + InfoNCE
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.2560 | Val mAP: 0.1271 | Mi-F1: 0.4215 | Ma-F1: 0.0461
Epoch [02/35] Loss: 0.1912 | Val mAP: 0.1641 | Mi-F1: 0.4752 | Ma-F1: 0.0712
Epoch [03/35] Loss: 0.1723 | Val mAP: 0.1905 | Mi-F1: 0.5101 | Ma-F1: 0.0895
Epoch [04/35] Loss: 0.1608 | Val mAP: 0.2065 | Mi-F1: 0.5196 | Ma-F1: 0.1001
Epoch [05/35] Loss: 0.1532 | Val mAP: 0.2173 | Mi-F1: 0.5285 | Ma-F1: 0.1125
Epoch [06/35] Loss: 0.1482 | Val mAP: 0.2233 | Mi-F1: 0.5386 | Ma-F1: 0.1159
Epoch [07/35] Loss: 0.1443 | Val mAP: 0.2327 | Mi-F1: 0.5483 | Ma-F1: 0.1226
Epoch [08/35] Loss: 0.1415 | Val mAP: 0.2378 | Mi-F1: 0.5506 | Ma-F1: 0.1287
Epoch [09/35] Loss: 0.1393 | Val mAP: 0.2443 | Mi-F1: 0.5591 | Ma-F1: 0.1351
Epoch [10/35] Loss: 0.1377 | Val mAP: 0.2482 | Mi-F1: 0.5523 | Ma-F1: 0.1353
Epoch [11/35] Loss: 0.1360 | Val mAP: 0.2540 | Mi-F1: 0.5571 | Ma-F1: 0.1429
Epoch [12/35] Loss: 0.1345 | Val mAP: 0.2557 | Mi-F1: 0.5698 | Ma-F1: 0.1580
Epoch [13/35] Loss: 0.1335 | Val mAP: 0.2593 | Mi-F1: 0.5652 | Ma-F1: 0.1518
Epoch [14/35] Loss: 0.1325 | Val mAP: 0.2626 | Mi-F1: 0.5668 | Ma-F1: 0.1573
Epoch [15/35] Loss: 0.1313 | Val mAP: 0.2649 | Mi-F1: 0.5705 | Ma-F1: 0.1549
Epoch [16/35] Loss: 0.1305 | Val mAP: 0.2684 | Mi-F1: 0.5723 | Ma-F1: 0.1653
Epoch [17/35] Loss: 0.1296 | Val mAP: 0.2690 | Mi-F1: 0.5737 | Ma-F1: 0.1694
Epoch [18/35] Loss: 0.1289 | Val mAP: 0.2714 | Mi-F1: 0.5788 | Ma-F1: 0.1723
Epoch [19/35] Loss: 0.1283 | Val mAP: 0.2719 | Mi-F1: 0.5770 | Ma-F1: 0.1689
Epoch [20/35] Loss: 0.1276 | Val mAP: 0.2739 | Mi-F1: 0.5800 | Ma-F1: 0.1682
Epoch [21/35] Loss: 0.1270 | Val mAP: 0.2761 | Mi-F1: 0.5783 | Ma-F1: 0.1775
Epoch [22/35] Loss: 0.1266 | Val mAP: 0.2780 | Mi-F1: 0.5807 | Ma-F1: 0.1774
Epoch [23/35] Loss: 0.1260 | Val mAP: 0.2800 | Mi-F1: 0.5776 | Ma-F1: 0.1757
Epoch [24/35] Loss: 0.1257 | Val mAP: 0.2820 | Mi-F1: 0.5824 | Ma-F1: 0.1780
Epoch [25/35] Loss: 0.1251 | Val mAP: 0.2822 | Mi-F1: 0.5823 | Ma-F1: 0.1811
Epoch [26/35] Loss: 0.1247 | Val mAP: 0.2822 | Mi-F1: 0.5853 | Ma-F1: 0.1869
Epoch [27/35] Loss: 0.1243 | Val mAP: 0.2848 | Mi-F1: 0.5849 | Ma-F1: 0.1855
Epoch [28/35] Loss: 0.1238 | Val mAP: 0.2844 | Mi-F1: 0.5841 | Ma-F1: 0.1873
Epoch [29/35] Loss: 0.1235 | Val mAP: 0.2845 | Mi-F1: 0.5800 | Ma-F1: 0.1872
Epoch [30/35] Loss: 0.1232 | Val mAP: 0.2860 | Mi-F1: 0.5874 | Ma-F1: 0.1883
Epoch [31/35] Loss: 0.1229 | Val mAP: 0.2884 | Mi-F1: 0.5877 | Ma-F1: 0.1934
Epoch [32/35] Loss: 0.1224 | Val mAP: 0.2898 | Mi-F1: 0.5875 | Ma-F1: 0.1995
Epoch [33/35] Loss: 0.1222 | Val mAP: 0.2889 | Mi-F1: 0.5882 | Ma-F1: 0.1963
Epoch [34/35] Loss: 0.1218 | Val mAP: 0.2900 | Mi-F1: 0.5882 | Ma-F1: 0.1973
Epoch [35/35] Loss: 0.1216 | Val mAP: 0.2913 | Mi-F1: 0.5891 | Ma-F1: 0.1990
[TEST SET] mAP: 0.3315 | Micro-F1: 0.5865 | Macro-F1: 0.1759

==================================================
Training: Gated Fusion (No InfoNCE)
==================================================
MLP hidden_dims=[1024, 512, 256], activation=gelu, dropout=0.3
Epoch [01/35] Loss: 0.0879 | Val mAP: 0.1671 | Mi-F1: 0.5080 | Ma-F1: 0.0802
Epoch [02/35] Loss: 0.0703 | Val mAP: 0.2083 | Mi-F1: 0.5284 | Ma-F1: 0.0974
Epoch [03/35] Loss: 0.0668 | Val mAP: 0.2302 | Mi-F1: 0.5487 | Ma-F1: 0.1197
Epoch [04/35] Loss: 0.0648 | Val mAP: 0.2440 | Mi-F1: 0.5605 | Ma-F1: 0.1367
Epoch [05/35] Loss: 0.0632 | Val mAP: 0.2553 | Mi-F1: 0.5713 | Ma-F1: 0.1535
Epoch [06/35] Loss: 0.0621 | Val mAP: 0.2592 | Mi-F1: 0.5822 | Ma-F1: 0.1664
Epoch [07/35] Loss: 0.0611 | Val mAP: 0.2695 | Mi-F1: 0.5831 | Ma-F1: 0.1691
Epoch [08/35] Loss: 0.0603 | Val mAP: 0.2737 | Mi-F1: 0.5866 | Ma-F1: 0.1759
Epoch [09/35] Loss: 0.0594 | Val mAP: 0.2797 | Mi-F1: 0.5837 | Ma-F1: 0.1793
Epoch [10/35] Loss: 0.0592 | Val mAP: 0.2828 | Mi-F1: 0.5870 | Ma-F1: 0.1898
Epoch [11/35] Loss: 0.0581 | Val mAP: 0.2859 | Mi-F1: 0.5888 | Ma-F1: 0.1883
Epoch [12/35] Loss: 0.0574 | Val mAP: 0.2878 | Mi-F1: 0.5928 | Ma-F1: 0.1938
Epoch [13/35] Loss: 0.0569 | Val mAP: 0.2882 | Mi-F1: 0.5906 | Ma-F1: 0.2018
Epoch [14/35] Loss: 0.0564 | Val mAP: 0.2897 | Mi-F1: 0.5939 | Ma-F1: 0.2010
Epoch [15/35] Loss: 0.0558 | Val mAP: 0.2918 | Mi-F1: 0.5930 | Ma-F1: 0.2053
Epoch [16/35] Loss: 0.0552 | Val mAP: 0.2933 | Mi-F1: 0.5957 | Ma-F1: 0.2077
Epoch [17/35] Loss: 0.0548 | Val mAP: 0.2941 | Mi-F1: 0.5966 | Ma-F1: 0.2111
Epoch [18/35] Loss: 0.0542 | Val mAP: 0.2951 | Mi-F1: 0.5958 | Ma-F1: 0.2139
Epoch [19/35] Loss: 0.0539 | Val mAP: 0.2969 | Mi-F1: 0.5954 | Ma-F1: 0.2110
Epoch [20/35] Loss: 0.0535 | Val mAP: 0.2965 | Mi-F1: 0.5969 | Ma-F1: 0.2217
Epoch [21/35] Loss: 0.0535 | Val mAP: 0.2962 | Mi-F1: 0.6006 | Ma-F1: 0.2252
Epoch [22/35] Loss: 0.0525 | Val mAP: 0.2977 | Mi-F1: 0.5957 | Ma-F1: 0.2304
Epoch [23/35] Loss: 0.0522 | Val mAP: 0.2980 | Mi-F1: 0.5962 | Ma-F1: 0.2335
Epoch [24/35] Loss: 0.0518 | Val mAP: 0.2966 | Mi-F1: 0.6017 | Ma-F1: 0.2374
Epoch [25/35] Loss: 0.0514 | Val mAP: 0.2970 | Mi-F1: 0.6007 | Ma-F1: 0.2392
Epoch [26/35] Loss: 0.0511 | Val mAP: 0.2961 | Mi-F1: 0.5984 | Ma-F1: 0.2324
Epoch [27/35] Loss: 0.0507 | Val mAP: 0.2967 | Mi-F1: 0.5949 | Ma-F1: 0.2347
Epoch [28/35] Loss: 0.0508 | Val mAP: 0.2964 | Mi-F1: 0.5992 | Ma-F1: 0.2380
Epoch [29/35] Loss: 0.0500 | Val mAP: 0.2979 | Mi-F1: 0.6020 | Ma-F1: 0.2441
Epoch [30/35] Loss: 0.0498 | Val mAP: 0.2959 | Mi-F1: 0.6004 | Ma-F1: 0.2415
Epoch [31/35] Loss: 0.0497 | Val mAP: 0.2973 | Mi-F1: 0.5986 | Ma-F1: 0.2514
Epoch [32/35] Loss: 0.0491 | Val mAP: 0.2950 | Mi-F1: 0.5990 | Ma-F1: 0.2478
Epoch [33/35] Loss: 0.0488 | Val mAP: 0.2955 | Mi-F1: 0.5993 | Ma-F1: 0.2473
Epoch [34/35] Loss: 0.0488 | Val mAP: 0.2948 | Mi-F1: 0.5970 | Ma-F1: 0.2483
Epoch [35/35] Loss: 0.0483 | Val mAP: 0.2952 | Mi-F1: 0.5948 | Ma-F1: 0.2483
[TEST SET] mAP: 0.3474 | Micro-F1: 0.6047 | Macro-F1: 0.2365

====================================================================
Experiment Model               | mAP (%)  | Micro-F1 | Macro-F1
--------------------------------------------------------------------
Baseline_A (BoW)               | 17.46     | 40.92     | 9.40
Baseline_B (Color+Tex)         | 34.90     | 60.18     | 22.88
Early_Fusion (Concat)          | 35.16     | 60.72     | 24.28
Gated_Fusion + InfoNCE         | 33.15     | 58.65     | 17.59
Gated_Fusion (No NCE)          | 34.74     | 60.47     | 23.65
====================================================================
```

## Analysis

The BCE-based experiments show that **Early Fusion (Concat)** achieves the best overall performance on the 81-label task:

| Model | mAP (%) | Micro-F1 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Baseline_A (BoW) | 17.46 | 40.92 | 9.40 |
| Baseline_B (Color+Texture) | 34.90 | 60.18 | 22.88 |
| Early_Fusion (Concat) | **35.16** | **60.72** | **24.28** |
| Gated_Fusion + InfoNCE | 33.15 | 58.65 | 17.59 |
| Gated_Fusion (No InfoNCE) | 34.74 | 60.47 | 23.65 |

### Main Findings

First, the **Color+Texture** features are much stronger than the **BoW** features. The BoW-only baseline obtains only 17.46% mAP and 9.40% Macro-F1, while the Color+Texture baseline reaches 34.90% mAP and 22.88% Macro-F1. This suggests that global visual descriptors such as color and texture are more discriminative for this 81-label classification task than the BoW representation alone.

Second, feature fusion is useful, but the improvement is relatively small. Early Fusion improves mAP from 34.90% to 35.16% compared with the Color+Texture baseline. The gain is more visible in Macro-F1, which increases from 22.88% to 24.28%. This indicates that BoW may still provide complementary information for some difficult or less frequent labels, even though it is weak as a standalone feature.

Third, **InfoNCE hurts performance in this setting**. Gated Fusion without InfoNCE achieves 34.74% mAP and 23.65% Macro-F1, while Gated Fusion with InfoNCE drops to 33.15% mAP and 17.59% Macro-F1. The decrease in Macro-F1 is especially large. A likely explanation is that BoW and Color+Texture are heterogeneous feature types. Forcing them to align through InfoNCE may weaken their modality-specific information instead of improving complementarity.

Fourth, several models show signs of overfitting. For example, the BoW baseline reaches its best validation mAP around epoch 11, but the validation mAP decreases by epoch 35 while the training loss keeps going down. Early Fusion also peaks around the middle or later part of training and then slightly declines. This suggests that early stopping based on validation mAP would likely produce better final test results than always using the last epoch.

### Conclusion

Among the BCE-based baselines, **Early Fusion (Concat)** is the strongest method. The results suggest that Color+Texture features dominate the prediction performance, while BoW features provide limited but useful complementary information. However, the current InfoNCE-based alignment strategy is not beneficial for this task. For future experiments, it would be useful to compare BCE, Focal Loss, and ASL under the same fusion settings, especially using Macro-F1 as an important metric because the 81-label task is likely class-imbalanced.
