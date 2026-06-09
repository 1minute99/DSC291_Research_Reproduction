## Table 1 - Faithfulness to true class (mean BERTScore-F1 vs ground-truth class)

| model / dataset                  | DnD     | MILAN       | NetDissect    |
|:---------------------------------|:--------|:------------|:--------------|
| resnet50 / imagenet (fc, n=1000) | 0.195   | 0.122       | 0.169         |
| resnet18 / places365 (fc)        | no data | no fc layer | 1 method only |
| alexnet / any                    | no data | no data     | no data       |
| resnet18 / imagenet              | no data | no data     | no data       |
| resnet50 / places365             | no data | no data     | no data       |

## Table 2 - Per-class faithfulness (6 ImageNet subclasses), resnet50/imagenet

| class                    |   MILAN |    DnD |   NetDissect |
|:-------------------------|--------:|-------:|-------------:|
| tench                    |   0.013 | -0.101 |        0.078 |
| English springer spaniel |   0.043 | -0.009 |       -0.018 |
| cassette player          |   0.146 |  0.249 |        0.145 |
| church                   |   0.092 |  0.635 |        0.382 |
| garbage truck            |   0.326 |  0.308 |        0.204 |
| golf ball                |   0.127 |  0.678 |       -0.051 |
| MEAN                     |   0.124 |  0.293 |        0.123 |

## Table 3a - RAW: resnet50 / imagenet, fc (class) neurons

| class_name               | DnD                                            | MILAN                 | NetDissect       |
|:-------------------------|:-----------------------------------------------|:----------------------|:-----------------|
| tench                    | fishing and catching fish                      | Living things         | scaly            |
| English springer spaniel | various dog scenes                             | Dogs                  | dog              |
| cassette player          | audio equipment                                | Electronics           | music studio     |
| church                   | church architecture                            | The tops of buildings | cathedral indoor |
| garbage truck            | vehicles for transportation and waste disposal | Vehicles              | weighbridge      |
| golf ball                | golf ball variations                           | Round objects         | fairway          |

## Table 3b - RAW: resnet18 / places365, layer4 (feature neurons; no class label)

|   unit | MILAN                 | NetDissect   | CLIP-Dissect   |
|-------:|:----------------------|:-------------|:---------------|
|      0 | Counters              | reception    | kitchen        |
|      1 | Shelves and books     | shop window  | shops          |
|      2 | Water                 | water        | pizza          |
|      3 | Green colored objects | auditorium   | auditorium     |
|      4 | Buildings             | dining room  | lobby          |

## Table 3c - RAW: resnet152 / imagenet, layer4 (feature neurons; no class label)

|   unit | DnD                                     | MILAN                 | CLIP-Dissect   |
|-------:|:----------------------------------------|:----------------------|:---------------|
|   1039 | outdoor activities with objects         | People                | helmet         |
|   1772 | nature and animals                      | Ice cream and animals | lotion         |
|   1602 | food-related scenes                     | Food                  | wraps          |
|   1915 | musical instruments and household items | Circular objects      | guitar         |
|   1592 | scenes with sky and nature elements     | Towers                | landmarks      |
